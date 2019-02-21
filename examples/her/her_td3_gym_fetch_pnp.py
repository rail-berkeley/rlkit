import gym

import rlkit.torch.pytorch_util as ptu
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import (
    GaussianAndEpsilonStrategy
)
from rlkit.torch.her.her import HerTd3
import rlkit.samplers.rollout_functions as rf


from rlkit.torch.networks import FlattenMlp, MlpPolicy, QNormalizedFlattenMlp, CompositeNormalizedMlpPolicy
from rlkit.torch.data_management.normalizer import CompositeNormalizer


def experiment(variant):
    try:
        import robotics_recorder
    except ImportError as e:
        print(e)

    env = gym.make(variant['env_id'])
    es = GaussianAndEpsilonStrategy(
        action_space=env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.3,
    )
    obs_dim = env.observation_space.spaces['observation'].low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    action_dim = env.action_space.low.size

    shared_normalizer = CompositeNormalizer(obs_dim + goal_dim, action_dim, obs_clip_range=5)

    qf1 = QNormalizedFlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
        composite_normalizer=shared_normalizer
    )
    qf2 = QNormalizedFlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
        composite_normalizer=shared_normalizer
    )
    import torch
    policy = CompositeNormalizedMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
        composite_normalizer=shared_normalizer,
        output_activation=torch.tanh
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    algorithm = HerTd3(
        her_kwargs=dict(
            observation_key='observation',
            desired_goal_key='desired_goal'
        ),
        td3_kwargs = dict(
            env=env,
            qf1=qf1,
            qf2=qf2,
            policy=policy,
            exploration_policy=exploration_policy
        ),
        replay_buffer=replay_buffer,
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
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=5000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=500,
            max_path_length=50,
            batch_size=128,
            discount=0.98,
            save_algorithm=True,
        ),
        replay_buffer_kwargs=dict(
            max_size=100000,
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0.0,
        ),
        render=False,
        env_id="FetchPickAndPlace-v1",
        doodad_docker_image="", # Set
        gpu_doodad_docker_image="", # Set
        save_video=False,
        save_video_period=50,
    )

    from rlkit.launchers.launcher_util import run_experiment

    run_experiment(
        experiment,
        exp_prefix="her_td3_gym_fetch_pnp_test",  # Make sure no spaces...
        region="us-east-2",
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True, # Note: online normalization is very slow without GPU.
        spot_price=.5,
        snapshot_mode='gap_and_last',
        snapshot_gap=100,
        num_exps_per_instance=2
    )

