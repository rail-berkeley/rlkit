from gym.envs.mujoco import (
    HalfCheetahEnv,
    AntEnv,
    Walker2dEnv,
    InvertedDoublePendulumEnv,
    HopperEnv,
    HumanoidEnv,
    SwimmerEnv,
)
from gym.envs.classic_control import PendulumEnv
import gym

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv, StackObservationEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)

from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.visualization.video import save_paths

ENV_PARAMS = {
    'half-cheetah': {  # 6 DoF
        'env_class': HalfCheetahEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 1000,
    },
    'hopper': {  # 6 DoF
        'env_class': HopperEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 1000,
    },
    'humanoid': {  # 6 DoF
        'env_class': HumanoidEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 3000,
    },
    'inv-double-pendulum': {  # 2 DoF
        'env_class': InvertedDoublePendulumEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 100,
    },
    'pendulum': {  # 2 DoF
        'env_class': PendulumEnv,
        'num_expl_steps_per_train_loop': 200,
        'max_path_length': 200,
        'num_epochs': 200,
        'min_num_steps_before_training': 2000,
        'target_update_period': 200,
    },
    'ant': {  # 6 DoF
        'env_class': AntEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 3000,
    },
    'walker': {  # 6 DoF
        'env_class': Walker2dEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 3000,
    },
    'swimmer': {  # 6 DoF
        'env_class': SwimmerEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 2000,
    },

    'pen-v0': {
        'env_id': 'pen-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 200,
        # 'num_epochs': 1000,
    },
    'door-v0': {
        'env_id': 'door-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 200,
        # 'num_epochs': 1000,
    },
    'relocate-v0': {
        'env_id': 'relocate-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 200,
        # 'num_epochs': 1000,
    },
    'hammer-v0': {
        'env_id': 'hammer-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 200,
        # 'num_epochs': 1000,
    },
}


def experiment(variant):
    env_params = ENV_PARAMS[variant['env']]
    variant.update(env_params)

    if 'env_id' in env_params:

        expl_env = gym.make(env_params['env_id'])
        eval_env = gym.make(env_params['env_id'])
    else:
        expl_env = NormalizedBoxEnv(variant['env_class']())
        eval_env = NormalizedBoxEnv(variant['env_class']())

    path_loader_kwargs = variant.get("path_loader_kwargs", {})
    stack_obs = path_loader_kwargs.get("stack_obs", 1)
    expl_env = StackObservationEnv(expl_env, stack_obs=stack_obs)
    eval_env = StackObservationEnv(eval_env, stack_obs=stack_obs)

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    if hasattr(expl_env, 'info_sizes'):
        env_info_sizes = expl_env.info_sizes
    else:
        env_info_sizes = dict()

    replay_buffer_kwargs=dict(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )


    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs'],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    replay_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )
    trainer = AWACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    if variant['collection_mode'] == 'online':
        expl_path_collector = MdpStepCollector(
            expl_env,
            policy,
        )
        algorithm = TorchOnlineRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            max_path_length=variant['max_path_length'],
            batch_size=variant['batch_size'],
            num_epochs=variant['num_epochs'],
            num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
            num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
            num_trains_per_train_loop=variant['num_trains_per_train_loop'],
            min_num_steps_before_training=variant['min_num_steps_before_training'],
        )
    else:
        if variant.get("deterministic_exploration", False):
            expl_policy = eval_policy
        else:
            expl_policy = policy
        expl_path_collector = MdpPathCollector(
            expl_env,
            expl_policy,
        )
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            max_path_length=variant['max_path_length'],
            batch_size=variant['batch_size'],
            num_epochs=variant['num_epochs'],
            num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
            num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
            num_trains_per_train_loop=variant['num_trains_per_train_loop'],
            min_num_steps_before_training=variant['min_num_steps_before_training'],
        )
    algorithm.to(ptu.device)

    demo_train_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )
    demo_test_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )

    if variant.get('save_paths', False):
        algorithm.post_train_funcs.append(save_paths)

    if variant.get('load_demos', False):
        path_loader_class = variant.get('path_loader_class', MDPPathLoader)
        path_loader = path_loader_class(trainer,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            **path_loader_kwargs
        )
        path_loader.load_demos()
    if variant.get('pretrain_policy', False):
        trainer.pretrain_policy_with_bc()
    if variant.get('pretrain_rl', False):
        trainer.pretrain_q_with_bc_data()

    algorithm.train()
