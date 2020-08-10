"""
This should results in an average return of -20 by the end of training.

Usually hits -30 around epoch 50.
Note that one epoch = 5k steps, so 200 epochs = 1 million steps.
"""
import gym

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import GoalConditionedPathCollector

from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

try:
    import multiworld.envs.gridworlds
except ImportError as e:
    print("To run this example, you need to install `multiworld`. See "
          "https://github.com/vitchyr/multiworld.")
    raise e


def experiment(variant):
    expl_env = gym.make('GoalGridworld-v0')
    eval_env = gym.make('GoalGridworld-v0')

    obs_dim = expl_env.observation_space.spaces['observation'].low.size
    goal_dim = expl_env.observation_space.spaces['desired_goal'].low.size
    action_dim = expl_env.action_space.n
    qf = ConcatMlp(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    target_qf = ConcatMlp(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    eval_policy = ArgmaxDiscretePolicy(qf)
    exploration_strategy = EpsilonGreedy(
        action_space=expl_env.action_space,
    )
    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=exploration_strategy,
        policy=eval_policy,
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        **variant['replay_buffer_kwargs']
    )
    observation_key = 'observation'
    desired_goal_key = 'desired_goal'
    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        eval_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = GoalConditionedPathCollector(
        expl_env,
        expl_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
        **variant['trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            max_path_length=50,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
            batch_size=128,
        ),
        trainer_kwargs=dict(
            discount=0.99,
        ),
        replay_buffer_kwargs=dict(
            max_size=100000,
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0.0,
        ),
    )
    setup_logger('her-dqn-gridworld-experiment', variant=variant)
    experiment(variant)
