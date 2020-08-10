"""
Example of running PyTorch implementation of DDPG on HalfCheetah.
"""
import copy

from gym.envs.mujoco import HalfCheetahEnv

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.torch.ddpg.ddpg import DDPGTrainer
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def experiment(variant):
    eval_env = NormalizedBoxEnv(HalfCheetahEnv())
    expl_env = NormalizedBoxEnv(HalfCheetahEnv())
    # Or for a specific version:
    # import gym
    # env = NormalizedBoxEnv(gym.make('HalfCheetah-v1'))
    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    qf = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_qf = copy.deepcopy(qf)
    target_policy = copy.deepcopy(policy)
    eval_path_collector = MdpPathCollector(eval_env, policy)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=OUStrategy(action_space=expl_env.action_space),
        policy=policy,
    )
    expl_path_collector = MdpPathCollector(expl_env, exploration_policy)
    replay_buffer = EnvReplayBuffer(variant['replay_buffer_size'], expl_env)
    trainer = DDPGTrainer(
        qf=qf,
        target_qf=target_qf,
        policy=policy,
        target_policy=target_policy,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=10000,
            max_path_length=1000,
            batch_size=128,
        ),
        trainer_kwargs=dict(
            use_soft_update=True,
            tau=1e-2,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        replay_buffer_size=int(1E6),
    )
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    setup_logger('name-of-experiment', variant=variant)
    experiment(variant)
