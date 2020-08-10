import gym
# import roboverse
# from rlkit.data_management.awr_env_replay_buffer import AWREnvReplayBuffer
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.split_buffer import SplitReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv, StackObservationEnv, RewardWrapperEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector import MdpPathCollector, ObsDictPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)

from rlkit.demos.source.hdf5_path_loader import HDF5PathLoader
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
# from rlkit.visualization.video import save_paths, VideoSaveFunction

from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.core.image_env import ImageEnv
from multiworld.core.gym_to_multi_env import GymToMultiEnv

import torch
import numpy as np
from torchvision.utils import save_image

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import GaussianAndEpsilonStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy

import os.path as osp
from rlkit.core import logger
from rlkit.util.io import load_local_or_remote_file
import pickle

# from rlkit.envs.images import Renderer, InsertImageEnv, EnvRenderer
from rlkit.envs.make_env import make

ENV_PARAMS = {
    'HalfCheetah-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/hc_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/hc_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'Ant-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/ant_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/ant_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'Walker2d-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/walker_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/walker_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },

    'SawyerRigGrasp-v0': {
        'env_id': 'SawyerRigGrasp-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 50,
        # 'num_epochs': 1000,
    },

    'pen-binary-v0': {
        'env_id': 'pen-binary-v0',
        'max_path_length': 200,
        'env_demo_path': dict(
            path="demos/icml2020/hand/pen2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/hand/pen_bc_sparse4.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'door-binary-v0': {
        'env_id': 'door-binary-v0',
        'max_path_length': 200,
        'env_demo_path': dict(
            path="demos/icml2020/hand/door2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/hand/door_bc_sparse4.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'relocate-binary-v0': {
        'env_id': 'relocate-binary-v0',
        'max_path_length': 200,
        'env_demo_path': dict(
            path="demos/icml2020/hand/relocate2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/hand/relocate_bc_sparse4.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
}


def resume(variant):
    data = load_local_or_remote_file(variant.get("pretrained_algorithm_path"), map_location="cuda")
    algo = data['algorithm']

    algo.num_epochs = variant['num_epochs']

    post_pretrain_hyperparams = variant["trainer_kwargs"].get("post_pretrain_hyperparams", {})
    algo.trainer.set_algorithm_weights(**post_pretrain_hyperparams)

    algo.train()


def process_args(variant):
    if variant.get("debug", False):
        variant['max_path_length'] = 50
        variant['batch_size'] = 5
        variant['num_epochs'] = 5
        # variant['num_eval_steps_per_epoch'] = 100
        # variant['num_expl_steps_per_train_loop'] = 100
        variant['num_trains_per_train_loop'] = 10
        # variant['min_num_steps_before_training'] = 100
        variant['trainer_kwargs']['bc_num_pretrain_steps'] = min(10, variant['trainer_kwargs'].get('bc_num_pretrain_steps', 0))
        variant['trainer_kwargs']['q_num_pretrain1_steps'] = min(10, variant['trainer_kwargs'].get('q_num_pretrain1_steps', 0))
        variant['trainer_kwargs']['q_num_pretrain2_steps'] = min(10, variant['trainer_kwargs'].get('q_num_pretrain2_steps', 0))


def experiment(variant):
    if variant.get("pretrained_algorithm_path", False):
        resume(variant)
        return

    normalize_env = variant.get('normalize_env', True)
    env_id = variant.get('env_id', None)
    env_params = ENV_PARAMS.get(env_id, {})
    variant.update(env_params)
    env_class = variant.get('env_class', None)
    env_kwargs = variant.get('env_kwargs', {})

    expl_env = make(env_id, env_class, env_kwargs, normalize_env)
    eval_env = make(env_id, env_class, env_kwargs, normalize_env)

    if variant.get('add_env_demos', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_demo_path"])
    if variant.get('add_env_offpolicy_data', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_offpolicy_data_path"])

    path_loader_kwargs = variant.get("path_loader_kwargs", {})
    stack_obs = path_loader_kwargs.get("stack_obs", 1)
    if stack_obs > 1:
        expl_env = StackObservationEnv(expl_env, stack_obs=stack_obs)
        eval_env = StackObservationEnv(eval_env, stack_obs=stack_obs)

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    if hasattr(expl_env, 'info_sizes'):
        env_info_sizes = expl_env.info_sizes
    else:
        env_info_sizes = dict()

    qf_kwargs = variant.get("qf_kwargs", {})
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )

    policy_class = variant.get("policy_class", TanhGaussianPolicy)
    policy_kwargs = variant['policy_kwargs']
    policy_path = variant.get("policy_path", False)
    if policy_path:
        policy = load_local_or_remote_file(policy_path)
    else:
        policy = policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **policy_kwargs,
        )
    buffer_policy_path = variant.get("buffer_policy_path", False)
    if buffer_policy_path:
        buffer_policy = load_local_or_remote_file(buffer_policy_path)
    else:
        buffer_policy_class = variant.get("buffer_policy_class", policy_class)
        buffer_policy = buffer_policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **variant.get("buffer_policy_kwargs", policy_kwargs),
        )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )

    expl_policy = policy
    exploration_kwargs =  variant.get('exploration_kwargs', {})
    if exploration_kwargs:
        if exploration_kwargs.get("deterministic_exploration", False):
            expl_policy = MakeDeterministic(policy)

        exploration_strategy = exploration_kwargs.get("strategy", None)
        if exploration_strategy is None:
            pass
        elif exploration_strategy == 'ou':
            es = OUStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        elif exploration_strategy == 'gauss_eps':
            es = GaussianAndEpsilonStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],  # constant sigma
                epsilon=0,
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        else:
            error

    main_replay_buffer_kwargs=dict(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )
    replay_buffer_kwargs = dict(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )

    replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        **main_replay_buffer_kwargs,
    )
    if variant.get('use_validation_buffer', False):
        train_replay_buffer = replay_buffer
        validation_replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
            **main_replay_buffer_kwargs,
        )
        replay_buffer = SplitReplayBuffer(train_replay_buffer, validation_replay_buffer, 0.9)

    trainer_class = variant.get("trainer_class", AWACTrainer)
    trainer = trainer_class(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        buffer_policy=buffer_policy,
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

    if variant.get("save_video", False):
        if variant.get("presampled_goals", None):
            variant['image_env_kwargs']['presampled_goals'] = load_local_or_remote_file(variant['presampled_goals']).item()

        def get_img_env(env):
            renderer = EnvRenderer(**variant["renderer_kwargs"])
            img_env = InsertImageEnv(GymToMultiEnv(env), renderer=renderer)

        image_eval_env = ImageEnv(GymToMultiEnv(eval_env), **variant["image_env_kwargs"])
        # image_eval_env = get_img_env(eval_env)
        image_eval_path_collector = ObsDictPathCollector(
            image_eval_env,
            eval_policy,
            observation_key="state_observation",
        )
        image_expl_env = ImageEnv(GymToMultiEnv(expl_env), **variant["image_env_kwargs"])
        # image_expl_env = get_img_env(expl_env)
        image_expl_path_collector = ObsDictPathCollector(
            image_expl_env,
            expl_policy,
            observation_key="state_observation",
        )
        video_func = VideoSaveFunction(
            image_eval_env,
            variant,
            image_expl_path_collector,
            image_eval_path_collector,
        )
        algorithm.post_train_funcs.append(video_func)
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
    if variant.get('load_env_dataset_demos', False):
        path_loader_class = variant.get('path_loader_class', HDF5PathLoader)
        path_loader = path_loader_class(trainer,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            **path_loader_kwargs
        )
        path_loader.load_demos(expl_env.get_dataset())
    if variant.get('save_initial_buffers', False):
        buffers = dict(
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
        )
        buffer_path = osp.join(logger.get_snapshot_dir(), 'buffers.p')
        pickle.dump(buffers, open(buffer_path, "wb"))
    if variant.get('pretrain_buffer_policy', False):
        trainer.pretrain_policy_with_bc(
            buffer_policy,
            replay_buffer.train_replay_buffer,
            replay_buffer.validation_replay_buffer,
            10000,
            label="buffer",
        )
    if variant.get('pretrain_policy', False):
        trainer.pretrain_policy_with_bc(
            policy,
            demo_train_buffer,
            demo_test_buffer,
            trainer.bc_num_pretrain_steps,
        )
    if variant.get('pretrain_rl', False):
        trainer.pretrain_q_with_bc_data()
    if variant.get('save_pretrained_algorithm', False):
        p_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.p')
        pt_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.pt')
        data = algorithm._get_snapshot()
        data['algorithm'] = algorithm
        torch.save(data, open(pt_path, "wb"))
        torch.save(data, open(p_path, "wb"))
    if variant.get('train_rl', True):
        algorithm.train()
