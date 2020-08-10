import time
from multiworld.core.image_env import ImageEnv
from rlkit.core import logger
from rlkit.envs.vae_wrapper import temporary_mode

import cv2
import numpy as np
import os.path as osp

from rlkit.samplers.data_collector.vae_env import (
    VAEWrappedEnvPathCollector,
)
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.skewfit.online_vae_algorithm import OnlineVaeAlgorithm
from rlkit.util.io import load_local_or_remote_file
from rlkit.util.video import dump_video


def skewfit_full_experiment(variant):
    variant['skewfit_variant']['save_vae_data'] = True
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    skewfit_experiment(variant['skewfit_variant'])


def full_experiment_variant_preprocess(variant):
    train_vae_variant = variant['train_vae_variant']
    skewfit_variant = variant['skewfit_variant']
    if 'env_id' in variant:
        assert 'env_class' not in variant
        env_id = variant['env_id']
        skewfit_variant['env_id'] = env_id
        train_vae_variant['generate_vae_dataset_kwargs']['env_id'] = env_id
    else:
        env_class = variant['env_class']
        env_kwargs = variant['env_kwargs']
        train_vae_variant['generate_vae_dataset_kwargs']['env_class'] = (
            env_class
        )
        train_vae_variant['generate_vae_dataset_kwargs']['env_kwargs'] = (
            env_kwargs
        )
        skewfit_variant['env_class'] = env_class
        skewfit_variant['env_kwargs'] = env_kwargs
    init_camera = variant.get('init_camera', None)
    imsize = variant.get('imsize', 84)
    train_vae_variant['generate_vae_dataset_kwargs']['init_camera'] = (
        init_camera
    )
    train_vae_variant['generate_vae_dataset_kwargs']['imsize'] = imsize
    train_vae_variant['imsize'] = imsize
    skewfit_variant['imsize'] = imsize
    skewfit_variant['init_camera'] = init_camera


def train_vae_and_update_variant(variant):
    from rlkit.core import logger
    skewfit_variant = variant['skewfit_variant']
    train_vae_variant = variant['train_vae_variant']
    if skewfit_variant.get('vae_path', None) is None:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'vae_progress.csv', relative_to_snapshot_dir=True
        )
        vae, vae_train_data, vae_test_data = train_vae(train_vae_variant,
                                                       return_data=True)
        if skewfit_variant.get('save_vae_data', False):
            skewfit_variant['vae_train_data'] = vae_train_data
            skewfit_variant['vae_test_data'] = vae_test_data
        logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
        logger.remove_tabular_output(
            'vae_progress.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )
        skewfit_variant['vae_path'] = vae  # just pass the VAE directly
    else:
        if skewfit_variant.get('save_vae_data', False):
            vae_train_data, vae_test_data, info = generate_vae_dataset(
                train_vae_variant['generate_vae_dataset_kwargs']
            )
            skewfit_variant['vae_train_data'] = vae_train_data
            skewfit_variant['vae_test_data'] = vae_test_data


def train_vae(variant, return_data=False):
    from rlkit.util.ml_util import PiecewiseLinearSchedule
    from rlkit.torch.vae.conv_vae import (
        ConvVAE,
    )
    import rlkit.torch.vae.conv_vae as conv_vae
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    from rlkit.pythonplusplus import identity
    import torch
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    generate_vae_dataset_fctn = variant.get('generate_vae_data_fctn',
                                            generate_vae_dataset)
    train_data, test_data, info = generate_vae_dataset_fctn(
        variant['generate_vae_dataset_kwargs']
    )
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(
            **variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    if variant.get('decoder_activation', None) == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity
    architecture = variant['vae_kwargs'].get('architecture', None)
    if not architecture and variant.get('imsize') == 84:
        architecture = conv_vae.imsize84_default_architecture
    elif not architecture and variant.get('imsize') == 48:
        architecture = conv_vae.imsize48_default_architecture
    variant['vae_kwargs']['architecture'] = architecture
    variant['vae_kwargs']['imsize'] = variant.get('imsize')

    m = ConvVAE(
        representation_size,
        decoder_output_activation=decoder_activation,
        **variant['vae_kwargs']
    )
    m.to(ptu.device)
    t = ConvVAETrainer(train_data, test_data, m, beta=beta,
                       beta_schedule=beta_schedule, **variant['algo_kwargs'])
    save_period = variant['save_period']
    dump_skew_debug_plots = variant.get('dump_skew_debug_plots', False)
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(
            epoch,
            save_reconstruction=should_save_imgs,
            # save_vae=False,
        )
        if should_save_imgs:
            t.dump_samples(epoch)
        t.update_train_weights()
    logger.save_extra_data(m, 'vae.pkl', mode='pickle')
    if return_data:
        return m, train_data, test_data
    return m


def generate_vae_dataset(variant):
    env_class = variant.get('env_class', None)
    env_kwargs = variant.get('env_kwargs', None)
    env_id = variant.get('env_id', None)
    N = variant.get('N', 10000)
    test_p = variant.get('test_p', 0.9)
    use_cached = variant.get('use_cached', True)
    imsize = variant.get('imsize', 84)
    num_channels = variant.get('num_channels', 3)
    show = variant.get('show', False)
    init_camera = variant.get('init_camera', None)
    dataset_path = variant.get('dataset_path', None)
    oracle_dataset_using_set_to_goal = variant.get(
        'oracle_dataset_using_set_to_goal', False)
    random_rollout_data = variant.get('random_rollout_data', False)
    random_and_oracle_policy_data = variant.get('random_and_oracle_policy_data',
                                                False)
    random_and_oracle_policy_data_split = variant.get(
        'random_and_oracle_policy_data_split', 0)
    policy_file = variant.get('policy_file', None)
    n_random_steps = variant.get('n_random_steps', 100)
    vae_dataset_specific_env_kwargs = variant.get(
        'vae_dataset_specific_env_kwargs', None)
    save_file_prefix = variant.get('save_file_prefix', None)
    non_presampled_goal_img_is_garbage = variant.get(
        'non_presampled_goal_img_is_garbage', None)
    tag = variant.get('tag', '')
    from multiworld.core.image_env import ImageEnv, unormalize_image
    import rlkit.torch.pytorch_util as ptu
    info = {}
    if dataset_path is not None:
        dataset = load_local_or_remote_file(dataset_path)
        N = dataset.shape[0]
    else:
        if env_kwargs is None:
            env_kwargs = {}
        if save_file_prefix is None:
            save_file_prefix = env_id
        if save_file_prefix is None:
            save_file_prefix = env_class.__name__
        filename = "/tmp/{}_N{}_{}_imsize{}_random_oracle_split_{}{}.npy".format(
            save_file_prefix,
            str(N),
            init_camera.__name__ if init_camera else '',
            imsize,
            random_and_oracle_policy_data_split,
            tag,
        )
        if use_cached and osp.isfile(filename):
            dataset = np.load(filename)
            print("loaded data from saved file", filename)
        else:
            now = time.time()

            if env_id is not None:
                import gym
                import multiworld
                multiworld.register_all_envs()
                env = gym.make(env_id)
            else:
                if vae_dataset_specific_env_kwargs is None:
                    vae_dataset_specific_env_kwargs = {}
                for key, val in env_kwargs.items():
                    if key not in vae_dataset_specific_env_kwargs:
                        vae_dataset_specific_env_kwargs[key] = val
                env = env_class(**vae_dataset_specific_env_kwargs)
            if not isinstance(env, ImageEnv):
                env = ImageEnv(
                    env,
                    imsize,
                    init_camera=init_camera,
                    transpose=True,
                    normalize=True,
                    non_presampled_goal_img_is_garbage=non_presampled_goal_img_is_garbage,
                )
            else:
                imsize = env.imsize
                env.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage
            env.reset()
            info['env'] = env
            if random_and_oracle_policy_data:
                policy_file = load_local_or_remote_file(policy_file)
                policy = policy_file['policy']
                policy.to(ptu.device)
            if random_rollout_data:
                from rlkit.exploration_strategies.ou_strategy import OUStrategy
                policy = OUStrategy(env.action_space)
            dataset = np.zeros((N, imsize * imsize * num_channels),
                               dtype=np.uint8)
            for i in range(N):
                if random_and_oracle_policy_data:
                    num_random_steps = int(
                        N * random_and_oracle_policy_data_split)
                    if i < num_random_steps:
                        env.reset()
                        for _ in range(n_random_steps):
                            obs = env.step(env.action_space.sample())[0]
                    else:
                        obs = env.reset()
                        policy.reset()
                        for _ in range(n_random_steps):
                            policy_obs = np.hstack((
                                obs['state_observation'],
                                obs['state_desired_goal'],
                            ))
                            action, _ = policy.get_action(policy_obs)
                            obs, _, _, _ = env.step(action)
                elif oracle_dataset_using_set_to_goal:
                    print(i)
                    goal = env.sample_goal()
                    env.set_to_goal(goal)
                    obs = env._get_obs()
                elif random_rollout_data:
                    if i % n_random_steps == 0:
                        g = dict(
                            state_desired_goal=env.sample_goal_for_rollout())
                        env.set_to_goal(g)
                        policy.reset()
                        # env.reset()
                    u = policy.get_action_from_raw_action(
                        env.action_space.sample())
                    obs = env.step(u)[0]
                else:
                    env.reset()
                    for _ in range(n_random_steps):
                        obs = env.step(env.action_space.sample())[0]
                img = obs['image_observation']
                dataset[i, :] = unormalize_image(img)
                if show:
                    img = img.reshape(3, imsize, imsize).transpose()
                    img = img[::-1, :, ::-1]
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
                    # radius = input('waiting...')
            print("done making training data", filename, time.time() - now)
            np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info


def get_envs(variant):
    from multiworld.core.image_env import ImageEnv
    from rlkit.envs.vae_wrapper import VAEWrappedEnv
    from rlkit.util.io import load_local_or_remote_file

    render = variant.get('render', False)
    vae_path = variant.get("vae_path", None)
    reward_params = variant.get("reward_params", dict())
    init_camera = variant.get("init_camera", None)
    do_state_exp = variant.get("do_state_exp", False)
    presample_goals = variant.get('presample_goals', False)
    presample_image_goals_only = variant.get('presample_image_goals_only',
                                             False)
    presampled_goals_path = variant.get('presampled_goals_path', None)

    vae = load_local_or_remote_file(vae_path) if type(
        vae_path) is str else vae_path
    if 'env_id' in variant:
        import gym
        import multiworld
        multiworld.register_all_envs()
        env = gym.make(variant['env_id'])
    else:
        env = variant["env_class"](**variant['env_kwargs'])
    if not do_state_exp:
        if isinstance(env, ImageEnv):
            image_env = env
        else:
            image_env = ImageEnv(
                env,
                variant.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
            )
        if presample_goals:
            """
            This will fail for online-parallel as presampled_goals will not be
            serialized. Also don't use this for online-vae.
            """
            if presampled_goals_path is None:
                image_env.non_presampled_goal_img_is_garbage = True
                vae_env = VAEWrappedEnv(
                    image_env,
                    vae,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
                presampled_goals = variant['generate_goal_dataset_fctn'](
                    env=vae_env,
                    env_id=variant.get('env_id', None),
                    **variant['goal_generation_kwargs']
                )
                del vae_env
            else:
                presampled_goals = load_local_or_remote_file(
                    presampled_goals_path
                ).item()
            del image_env
            image_env = ImageEnv(
                env,
                variant.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
                presampled_goals=presampled_goals,
                **variant.get('image_env_kwargs', {})
            )
            vae_env = VAEWrappedEnv(
                image_env,
                vae,
                imsize=image_env.imsize,
                decode_goals=render,
                render_goals=render,
                render_rollouts=render,
                reward_params=reward_params,
                presampled_goals=presampled_goals,
                **variant.get('vae_wrapped_env_kwargs', {})
            )
            print("Presampling all goals only")
        else:
            vae_env = VAEWrappedEnv(
                image_env,
                vae,
                imsize=image_env.imsize,
                decode_goals=render,
                render_goals=render,
                render_rollouts=render,
                reward_params=reward_params,
                **variant.get('vae_wrapped_env_kwargs', {})
            )
            if presample_image_goals_only:
                presampled_goals = variant['generate_goal_dataset_fctn'](
                    image_env=vae_env.wrapped_env,
                    **variant['goal_generation_kwargs']
                )
                image_env.set_presampled_goals(presampled_goals)
                print("Presampling image goals only")
            else:
                print("Not using presampled goals")

        env = vae_env

    return env


def get_exploration_strategy(variant, env):
    from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
    from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
    from rlkit.exploration_strategies.ou_strategy import OUStrategy

    exploration_type = variant['exploration_type']
    exploration_noise = variant.get('exploration_noise', 0.1)
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            max_sigma=exploration_noise,
            min_sigma=exploration_noise,  # Constant sigma
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=exploration_noise,
            min_sigma=exploration_noise,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=exploration_noise,
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    return es


def skewfit_preprocess_variant(variant):
    if variant.get("do_state_exp", False):
        variant['observation_key'] = 'state_observation'
        variant['desired_goal_key'] = 'state_desired_goal'
        variant['achieved_goal_key'] = 'state_acheived_goal'


def skewfit_experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from rlkit.torch.networks import ConcatMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer

    skewfit_preprocess_variant(variant)
    env = get_envs(variant)

    uniform_dataset_fn = variant.get('generate_uniform_dataset_fn', None)
    if uniform_dataset_fn:
        uniform_dataset = uniform_dataset_fn(
            **variant['generate_uniform_dataset_kwargs']
        )
    else:
        uniform_dataset = None

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    hidden_sizes = variant.get('hidden_sizes', [400, 300])
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )

    vae = env.vae

    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=env.vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    vae_trainer = ConvVAETrainer(
        variant['vae_train_data'],
        variant['vae_test_data'],
        env.vae,
        **variant['online_vae_trainer_kwargs']
    )
    assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"
    max_path_length = variant['max_path_length']

    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['twin_sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = VAEWrappedEnvPathCollector(
        variant['evaluation_goal_sampling_mode'],
        env,
        MakeDeterministic(policy),
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = VAEWrappedEnvPathCollector(
        variant['exploration_goal_sampling_mode'],
        env,
        policy,
        max_path_length,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )

    algorithm = OnlineVaeAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        vae=vae,
        vae_trainer=vae_trainer,
        uniform_dataset=uniform_dataset,
        max_path_length=max_path_length,
        **variant['algo_kwargs']
    )

    if variant['custom_goal_sampler'] == 'replay_buffer':
        env.custom_goal_sampler = replay_buffer.sample_buffer_goals

    algorithm.to(ptu.device)
    vae.to(ptu.device)
    algorithm.train()


def get_video_save_func(rollout_function, env, policy, variant):
    logdir = logger.get_snapshot_dir()
    save_period = variant.get('save_video_period', 50)
    do_state_exp = variant.get("do_state_exp", False)
    dump_video_kwargs = variant.get("dump_video_kwargs", dict())
    if do_state_exp:
        imsize = variant.get('imsize')
        dump_video_kwargs['imsize'] = imsize
        image_env = ImageEnv(
            env,
            imsize,
            init_camera=variant.get('init_camera', None),
            transpose=True,
            normalize=True,
        )

        def save_video(algo, epoch):
            if epoch % save_period == 0 or epoch == algo.num_epochs:
                filename = osp.join(logdir,
                                    'video_{epoch}_env.mp4'.format(epoch=epoch))
                dump_video(image_env, policy, filename, rollout_function,
                           **dump_video_kwargs)
    else:
        image_env = env
        dump_video_kwargs['imsize'] = env.imsize

        def save_video(algo, epoch):
            if epoch % save_period == 0 or epoch == algo.num_epochs:
                filename = osp.join(logdir,
                                    'video_{epoch}_env.mp4'.format(epoch=epoch))
                temporary_mode(
                    image_env,
                    mode='video_env',
                    func=dump_video,
                    args=(image_env, policy, filename, rollout_function),
                    kwargs=dump_video_kwargs
                )
                filename = osp.join(logdir,
                                    'video_{epoch}_vae.mp4'.format(epoch=epoch))
                temporary_mode(
                    image_env,
                    mode='video_vae',
                    func=dump_video,
                    args=(image_env, policy, filename, rollout_function),
                    kwargs=dump_video_kwargs
                )
    return save_video
