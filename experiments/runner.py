import argparse
import json
from rlkit.launchers.launcher_util import run_experiment
parser = argparse.ArgumentParser()
parser.add_argument('--exp_prefix', type=str, default='')
parser.add_argument('--mode', type=str, default='local')
parser.add_argument('--variant', type=str)
parser.add_argument('--num_seeds', type=int, default=1)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--tmux_session_name', type=str, default='research')
args = parser.parse_args()
variant = json.loads(args.variant)


def experiment(variant):
	from autolab_core import YamlConfig
	from hrl_exp.envs.franka_lift import GymFrankaLiftVecEnv
	from hrl_exp.envs.wrappers import ImageEnvWrapper
	from rlkit.torch.model_based.dreamer.dreamer import DreamerTrainer
	from rlkit.torch.model_based.dreamer.dreamer_policy import DreamerPolicy, ActionSpaceSamplePolicy
	from rlkit.torch.model_based.dreamer.episode_replay_buffer import EpisodeReplayBuffer
	from rlkit.torch.model_based.dreamer.mlp import Mlp
	from rlkit.torch.model_based.dreamer.models import WorldModel, ActorModel
	from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
	from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
	import torch
	import os
	import rlkit
	from os.path import join
	import pickle
	import rlkit.torch.pytorch_util as ptu

	ptu.set_gpu_mode(True, gpu_id=variant.get('torch_device', 0))
	rlkit_project_dir = join(os.path.dirname(rlkit.__file__), os.pardir)
	cfg_path = join(rlkit_project_dir, 'experiments/run_franka_lift.yaml')

	train_cfg = YamlConfig(cfg_path)
	train_cfg['franka']['workspace_limits']['ee_lower'] = variant['env_kwargs']['ee_lower']
	train_cfg['franka']['workspace_limits']['ee_upper'] = variant['env_kwargs']['ee_upper']
	train_cfg['scene']['n_envs'] = variant['env_kwargs']['n_train_envs']
	train_cfg['scene']['gym']['device']['compute'] = variant['env_kwargs']['compute_device']
	train_cfg['scene']['gym']['device']['graphics'] = variant['env_kwargs']['graphics_device']
	train_cfg['rews']['block_distance_to_lift'] = variant['env_kwargs']['block_distance_to_lift']
	train_cfg['env']['fixed_schema'] = variant['env_kwargs']['fixed_schema']
	train_cfg['env']['randomize_block_pose_on_reset'] = variant['env_kwargs']['randomize_block_pose_on_reset']

	train_cfg['pytorch_format'] = True
	train_cfg['flatten'] = True
	expl_env = GymFrankaLiftVecEnv(train_cfg, **train_cfg['env'])
	expl_env = ImageEnvWrapper(expl_env, train_cfg)

	eval_cfg = pickle.loads(pickle.dumps(train_cfg))
	eval_cfg['scene']['n_envs'] = variant['env_kwargs']['n_eval_envs']

	eval_env = GymFrankaLiftVecEnv(eval_cfg, **eval_cfg['env'])
	eval_env = ImageEnvWrapper(eval_env, eval_cfg)

	obs_dim = expl_env.observation_space.low.size
	action_dim = eval_env.action_space.low.size

	world_model = WorldModel(
		action_dim,
		**variant['model_kwargs'],
	)
	actor = ActorModel(
		[variant['model_kwargs']['model_hidden_size']]*4,
		variant['model_kwargs']['stochastic_state_size'] + variant['model_kwargs']['deterministic_state_size'],
		action_dim,
		hidden_activation=torch.nn.functional.elu,
		split_size=expl_env.wrapped_env.num_primitives,
		split_dist=variant['actor_kwargs']['split_dist'] and (not variant['env_kwargs']['fixed_schema'])
	)
	vf = Mlp(
		hidden_sizes=[variant['model_kwargs']['model_hidden_size']]*3,
		output_size=1,
		input_size=variant['model_kwargs']['stochastic_state_size'] + variant['model_kwargs']['deterministic_state_size'],
		hidden_activation=torch.nn.functional.elu,
	)

	expl_policy = DreamerPolicy(
		world_model,
		actor,
		obs_dim,
		action_dim,
		split_dist=variant['actor_kwargs']['split_dist'] and (not variant['env_kwargs']['fixed_schema']),
		split_size=expl_env.wrapped_env.num_primitives,
		exploration=True
	)
	eval_policy = DreamerPolicy(
		world_model,
		actor,
		obs_dim,
		action_dim,
		split_dist=variant['actor_kwargs'],
		split_size=expl_env.wrapped_env.num_primitives,
		exploration=False,
	)

	rand_policy = ActionSpaceSamplePolicy(expl_env)

	expl_path_collector = VecMdpPathCollector(
		expl_env,
		expl_policy,
	)

	eval_path_collector = VecMdpPathCollector(
		eval_env,
		eval_policy,
	)

	replay_buffer = EpisodeReplayBuffer(
		variant['replay_buffer_size'],
		expl_env,
		4,
		obs_dim,
		action_dim,
		replace=False
	)
	trainer = DreamerTrainer(
		env=eval_env,
		world_model=world_model,
		actor=actor,
		vf=vf,
		**variant['trainer_kwargs']
	)
	algorithm = TorchBatchRLAlgorithm(
		trainer=trainer,
		exploration_env=expl_env,
		evaluation_env=eval_env,
		exploration_data_collector=expl_path_collector,
		evaluation_data_collector=eval_path_collector,
		replay_buffer=replay_buffer,
		pretrain_policy=rand_policy,
		**variant['algorithm_kwargs'],
	)
	algorithm.to(ptu.device)
	algorithm.train()

for _ in range(args.num_seeds):
	run_experiment(
		experiment,
		exp_prefix=args.exp_prefix,
		mode=args.mode,
		variant=variant,
		use_gpu=True,
		snapshot_mode='none',
		gpu_id=args.gpu_id,
	)

import libtmux
server = libtmux.Server()
session = server.find_where({ "session_name": args.tmux_session_name })
window = session.attached_window
window.kill_window()