"""
This should results in an average return of ~3000 by the end of training.

Usually hits 3000 around epoch 80-100. Within a see, the performance will be
a bit noisy from one epoch to the next (occasionally dips dow to ~2000).

Note that one epoch = 5k steps, so 200 epochs = 1 million steps.
"""
import gym

import rlkit.torch.pytorch_util as ptu
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.her.her import HerTd3
from rlkit.torch.her.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
import multiworld.envs.mujoco


def experiment(variant):
    env = gym.make('FetchReach-v1')
    env = gym.make('SawyerReachXYEnv-v1')
    es = GaussianStrategy(
        action_space=env.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    obs_dim = env.observation_space.spaces['observation'].low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    action_dim = env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        achieved_goal_key='state_achieved_goal',
        desired_goal_key='state_desired_goal',
        **variant['replay_buffer_kwargs']
    )
    algorithm = HerTd3(
        env=env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            # num_epochs=200,
            # num_steps_per_epoch=5000,
            # num_steps_per_eval=10000,
            # max_path_length=100,
            num_epochs=20,
            num_steps_per_epoch=500,
            num_steps_per_eval=100,
            max_path_length=50,
            min_num_steps_before_training=1000,
            batch_size=100,
            discount=0.99,
        ),
        replay_buffer_kwargs=dict(
            max_size=100000,
            fraction_goals_rollout_goals=1.0,
            fraction_goals_env_goals=0.0,
        ),
    )
    setup_logger('name-of-td3-experiment', variant=variant)
    experiment(variant)
