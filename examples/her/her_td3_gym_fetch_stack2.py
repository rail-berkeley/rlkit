import gym

import rlkit.torch.pytorch_util as ptu

from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)

from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import (
    GaussianAndEpsilonStrategy
)

from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.her.her import HerTd3
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy

from rlkit.launchers.launcher_util import run_experiment

import gym_fetch_stack

def experiment(variant):
    env = gym.make('FetchStack2-v1')
    es = GaussianAndEpsilonStrategy(
        action_space=env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.3,
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
        **variant['replay_buffer_kwargs']
    )
    algorithm = HerTd3(
        her_kwargs={"observation_key": "observation",
                    "desired_goal_key": "desired_goal",

        },
        td3_kwargs={
            "env": env,
            "qf1": qf1,
            "qf2": qf2,
            "policy": policy,
            "exploration_policy": exploration_policy,
            "replay_buffer": replay_buffer
        },
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=50,
            batch_size=128,
            discount=0.99,
        ),
        replay_buffer_kwargs=dict(
            max_size=100000,
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0.0,
        ),
    )
    setup_logger('her-td3-fetch-experiment', variant=variant)
    run_experiment(
        experiment,
        exp_prefix="rlkit-her_td3_gym_fetch_stack2",
        mode='ec2',
        variant=variant,
        use_gpu=False,
        spot_price=.03,
        region="us-east-2"
    )