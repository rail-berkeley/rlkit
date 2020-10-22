import argparse
import json

from rlkit.launchers.launcher_util import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--exp_prefix", type=str, default="")
parser.add_argument("--mode", type=str, default="local")
parser.add_argument("--variant", type=str)
parser.add_argument("--num_seeds", type=int, default=1)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--tmux_session_name", type=str, default="")
args = parser.parse_args()
variant = json.loads(args.variant)


def experiment(variant):
    from autolab_core import YamlConfig
    from hrl_exp.envs.franka_hex_screw import GymFrankaHexScrewVecEnv
    from hrl_exp.envs.franka_lift import GymFrankaLiftVecEnv
    from hrl_exp.envs.franka_blocks import GymFrankaBlocksVecEnv
    from hrl_exp.envs.wrappers import ImageEnvWrapper
    from rlkit.torch.model_based.dreamer.dreamer import DreamerTrainer
    from rlkit.torch.model_based.dreamer.dreamer_policy import (
        DreamerPolicy,
        ActionSpaceSamplePolicy,
    )
    from rlkit.torch.model_based.dreamer.episode_replay_buffer import (
        EpisodeReplayBuffer,
    )
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

    rlkit_project_dir = join(os.path.dirname(rlkit.__file__), os.pardir)

    if variant["env_class"] == "hex_screw":
        env_class = GymFrankaHexScrewVecEnv
        cfg_path = join(rlkit_project_dir, "cfg/run_franka_hex_screw.yaml")
        train_cfg = YamlConfig(cfg_path)
        train_cfg["rews"]["target_screw_angle"] = variant["env_kwargs"][
            "target_screw_angle"
        ]
        train_cfg["rews"]["target_screw_angle_tol"] = variant["env_kwargs"][
            "target_screw_angle_tol"
        ]
    elif variant["env_class"] == "lift":
        env_class = GymFrankaLiftVecEnv
        cfg_path = join(rlkit_project_dir, "cfg/run_franka_lift.yaml")
        train_cfg = YamlConfig(cfg_path)
        train_cfg["rews"]["block_distance_to_lift"] = variant["env_kwargs"][
            "block_distance_to_lift"
        ]
        train_cfg["env"]["randomize_block_pose_on_reset"] = variant["env_kwargs"][
            "randomize_block_pose_on_reset"
        ]
    elif variant["env_class"] == "blocks":
        env_class = GymFrankaBlocksVecEnv
        cfg_path = join(rlkit_project_dir, "cfg/run_franka_blocks.yaml")
        train_cfg = YamlConfig(cfg_path)
        train_cfg["rews"]["block_stack_tol"] = variant["env_kwargs"]["block_stack_tol"]
        train_cfg["rews"]["num_blocks_to_stack"] = variant["env_kwargs"][
            "num_blocks_to_stack"
        ]
    else:
        raise EnvironmentError("Invalid env class provided")

    train_cfg["franka"]["workspace_limits"]["ee_lower"] = variant["env_kwargs"][
        "ee_lower"
    ]
    train_cfg["franka"]["workspace_limits"]["ee_upper"] = variant["env_kwargs"][
        "ee_upper"
    ]
    train_cfg["scene"]["n_envs"] = variant["env_kwargs"]["n_train_envs"]
    train_cfg["env"]["fixed_schema"] = variant["env_kwargs"]["fixed_schema"]
    train_cfg["pytorch_format"] = True
    train_cfg["flatten"] = True

    eval_cfg = pickle.loads(pickle.dumps(train_cfg))
    eval_cfg["scene"]["n_envs"] = variant["env_kwargs"]["n_eval_envs"]

    expl_env = env_class(train_cfg, **train_cfg["env"])
    expl_env = ImageEnvWrapper(expl_env, train_cfg)

    eval_env = env_class(eval_cfg, **eval_cfg["env"])
    eval_env = ImageEnvWrapper(eval_env, eval_cfg)

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    world_model = WorldModel(
        action_dim,
        **variant["model_kwargs"],
    )
    actor = ActorModel(
        [variant["model_kwargs"]["model_hidden_size"]] * 4,
        variant["model_kwargs"]["stochastic_state_size"]
        + variant["model_kwargs"]["deterministic_state_size"],
        action_dim,
        hidden_activation=torch.nn.functional.elu,
        split_size=expl_env.wrapped_env.num_primitives,
        split_dist=variant["actor_kwargs"]["split_dist"]
        and (not variant["env_kwargs"]["fixed_schema"]),
    )
    vf = Mlp(
        hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]] * 3,
        output_size=1,
        input_size=variant["model_kwargs"]["stochastic_state_size"]
        + variant["model_kwargs"]["deterministic_state_size"],
        hidden_activation=torch.nn.functional.elu,
    )

    expl_policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        split_dist=variant["actor_kwargs"]["split_dist"]
        and (not variant["env_kwargs"]["fixed_schema"]),
        split_size=expl_env.wrapped_env.num_primitives,
        exploration=True,
        expl_amount=variant.get("expl_amount", 0),
    )
    eval_policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        split_dist=variant["actor_kwargs"],
        split_size=expl_env.wrapped_env.num_primitives,
        exploration=False,
    )

    rand_policy = ActionSpaceSamplePolicy(expl_env)

    expl_path_collector = VecMdpPathCollector(
        expl_env,
        expl_policy,
        save_env_in_snapshot=False,
        env_params=train_cfg,
        env_class=env_class,
    )

    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False,
        env_params=eval_cfg,
        env_class=env_class,
    )

    replay_buffer = EpisodeReplayBuffer(
        variant["replay_buffer_size"],
        expl_env,
        variant["trainer_kwargs"]["imagination_horizon"],
        obs_dim,
        action_dim,
        replace=False,
    )
    trainer = DreamerTrainer(
        env=eval_env,
        world_model=world_model,
        actor=actor,
        vf=vf,
        **variant["trainer_kwargs"],
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        pretrain_policy=rand_policy,
        **variant["algorithm_kwargs"],
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
        snapshot_mode="last",
        gpu_id=args.gpu_id,
    )

if args.tmux_session_name:
    import libtmux

    server = libtmux.Server()
    session = server.find_where({"session_name": args.tmux_session_name})
    window = session.attached_window
    window.kill_window()
