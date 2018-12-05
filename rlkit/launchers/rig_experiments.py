import os.path as osp

import cv2
import numpy as np
import time

import rlkit.torch.vae.conv_vae as conv_vae
import torch
from rlkit.util.io import load_local_or_remote_file
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy

from rlkit.pythonplusplus import identity
from rlkit.envs.vae_wrapper import VAEWrappedEnv
from rlkit.torch.vae.conv_vae import ConvVAE
from rlkit.torch.vae.vae_trainer import ConvVAETrainer

import rlkit.samplers.rollout_functions as rf
import rlkit.torch.pytorch_util as ptu
from multiworld.core.image_env import ImageEnv, unormalize_image
from rlkit.core import logger
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.torch.her.her import HerTd3
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.util.video import dump_video
import gym
# trigger registration
# noinspection PyUnresolvedReferences
import multiworld.envs.pygame
# noinspection PyUnresolvedReferences
import multiworld.envs.mujoco


def grill_her_td3_full_experiment(variant):
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    grill_her_td3_experiment(variant['grill_variant'])


def full_experiment_variant_preprocess(variant):
    train_vae_variant = variant['train_vae_variant']
    grill_variant = variant['grill_variant']
    if 'env_id' in variant:
        assert 'env_class' not in variant
        env_id = variant['env_id']
        grill_variant['env_id'] = env_id
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
        grill_variant['env_class'] = env_class
        grill_variant['env_kwargs'] = env_kwargs
    init_camera = variant.get('init_camera', None)
    imsize = variant.get('imsize', 84)
    train_vae_variant['generate_vae_dataset_kwargs']['init_camera'] = (
        init_camera
    )
    train_vae_variant['generate_vae_dataset_kwargs']['imsize'] = imsize
    train_vae_variant['imsize'] = imsize
    grill_variant['imsize'] = imsize
    grill_variant['init_camera'] = init_camera


def train_vae_and_update_variant(variant):
    grill_variant = variant['grill_variant']
    train_vae_variant = variant['train_vae_variant']
    if grill_variant.get('vae_path', None) is None:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'vae_progress.csv', relative_to_snapshot_dir=True
        )
        vae, vae_train_data, vae_test_data = train_vae(
            train_vae_variant,
            return_data=True,
        )
        if grill_variant.get('save_vae_data', False):
            grill_variant['vae_train_data'] = vae_train_data
            grill_variant['vae_test_data'] = vae_test_data
        logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
        logger.remove_tabular_output(
            'vae_progress.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )
        grill_variant['vae_path'] = vae  # just pass the VAE directly
    else:
        if grill_variant.get('save_vae_data', False):
            vae_train_data, vae_test_data, info = generate_vae_dataset(
                train_vae_variant['generate_vae_dataset_kwargs']
            )
            grill_variant['vae_train_data'] = vae_train_data
            grill_variant['vae_test_data'] = vae_test_data


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
    oracle_dataset_from_policy = variant.get('oracle_dataset_from_policy',
                                             False)
    random_and_oracle_policy_data = variant.get('random_and_oracle_policy_data',
                                                False)
    random_and_oracle_policy_data_split = variant.get(
        'random_and_oracle_policy_data_split', 0)
    random_rollout_data = variant.get('random_rollout_data', False)
    policy_file = variant.get('policy_file', None)
    n_random_steps = variant.get('n_random_steps', 100)
    vae_dataset_specific_env_kwargs = variant.get(
        'vae_dataset_specific_env_kwargs', None)
    save_file_prefix = variant.get('save_file_prefix', None)
    non_presampled_goal_img_is_garbage = variant.get(
        'non_presampled_goal_img_is_garbage', None)
    tag = variant.get('tag', '')
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
            if oracle_dataset_from_policy or random_and_oracle_policy_data:
                policy_file = load_local_or_remote_file(policy_file)
                policy = policy_file['policy']
                policy.to(ptu.device)
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
                    goal = env.sample_goal()
                    env.set_to_goal(goal)
                    obs = env._get_obs()
                elif random_rollout_data:
                    if i % n_random_steps == 0:
                        g = env.sample_goal()
                        env.set_to_goal(g)
                    obs = env.step(env.action_space.sample())[0]
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


def train_vae(variant, return_data=False):
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    generate_vae_dataset_fctn = variant.get('generate_vae_data_fctn',
                                            generate_vae_dataset)
    train_data, test_data, info = generate_vae_dataset_fctn(
        variant['generate_vae_dataset_kwargs']
    )
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
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
                       **variant['algo_kwargs'])
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(
            epoch,
            save_reconstruction=should_save_imgs,
        )
        if should_save_imgs:
            t.dump_samples(epoch)
    logger.save_extra_data(m, 'vae.pkl', mode='pickle')
    if return_data:
        return m, train_data, test_data
    return m


def grill_her_td3_experiment(variant):
    env = get_envs(variant)
    es = get_exploration_strategy(variant, env)

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        env.observation_space.spaces[observation_key].low.size
        + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    td3_kwargs = algo_kwargs['td3_kwargs']
    td3_kwargs['training_env'] = env
    td3_kwargs['render'] = variant["render"]
    her_kwargs = algo_kwargs['her_kwargs']
    her_kwargs['observation_key'] = observation_key
    her_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = HerTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", True):
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            algorithm.eval_policy,
            variant,
        )
        algorithm.post_epoch_funcs.append(video_func)

    algorithm.to(ptu.device)
    env.vae.to(ptu.device)

    algorithm.train()


def get_envs(variant):

    render = variant.get('render', False)
    vae_path = variant.get("vae_path", None)
    reward_params = variant.get("reward_params", dict())
    init_camera = variant.get("init_camera", None)
    presample_goals = variant.get('presample_goals', False)
    presample_image_goals_only = variant.get('presample_image_goals_only',
                                             False)
    presampled_goals_path = variant.get('presampled_goals_path', None)

    vae = load_local_or_remote_file(vae_path) if type(
        vae_path) is str else vae_path
    if 'env_id' in variant:
        env = gym.make(variant['env_id'])
    else:
        env = variant["env_class"](**variant['env_kwargs'])

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

    training_mode = variant.get("training_mode", "train")
    testing_mode = variant.get("testing_mode", "test")
    env.add_mode('eval', testing_mode)
    env.add_mode('train', training_mode)
    env.add_mode('relabeling', training_mode)
    # relabeling_env.disable_render()
    env.add_mode("video_vae", 'video_vae')
    env.add_mode("video_env", 'video_env')
    return env


def get_exploration_strategy(variant, env):
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


def temporary_mode(env, mode, func, args=None, kwargs=None):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    cur_mode = env.cur_mode
    env.mode(env._mode_map[mode])
    return_val = func(*args, **kwargs)
    env.mode(cur_mode)
    return return_val
