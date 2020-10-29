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
    from hrl_exp.envs.mujoco_vec_wrappers import Async, make_env, VecEnv
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
    import rlkit.torch.pytorch_util as ptu
    from d4rl.kitchen.kitchen_envs import (
        KitchenKettleV0,
        KitchenLightSwitchV0,
        KitchenHingeCabinetV0,
        KitchenTopBurnerV0,
        KitchenSlideCabinetV0,
        KitchenMicrowaveV0,
    )

    env_class = variant["env_class"]
    env_kwargs = variant["env_kwargs"]
    if env_class == "microwave":
        env_class_ = KitchenMicrowaveV0
    elif env_class == "kettle":
        env_class_ = KitchenKettleV0
    elif env_class == "slide_cabinet":
        env_class_ = KitchenSlideCabinetV0
    elif env_class == "hinge_cabinet":
        env_class_ = KitchenHingeCabinetV0
    elif env_class == "top_burner":
        env_class_ = KitchenTopBurnerV0
    elif env_class == "light_switch":
        env_class_ = KitchenLightSwitchV0
    else:
        raise EnvironmentError("invalid env provided")
    expl_envs = [
        Async(
            lambda: make_env(
                env_class=env_class_,
                env_kwargs=variant["env_kwargs"],
            ),
            strategy="process",
        )
        for _ in range(variant["num_expl_envs"])
    ]

    eval_envs = [
        Async(
            lambda: make_env(
                env_class=env_class_,
                env_kwargs=variant["env_kwargs"],
            ),
            strategy="process",
        )
        for _ in range(variant["num_eval_envs"])
    ]

    expl_env = VecEnv(expl_envs)
    eval_env = VecEnv(eval_envs)

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
        exploration=True,
        expl_amount=variant.get("expl_amount", 0),
    )
    eval_policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        exploration=False,
    )

    rand_policy = ActionSpaceSamplePolicy(expl_env)

    expl_path_collector = VecMdpPathCollector(
        expl_env,
        expl_policy,
        save_env_in_snapshot=False,
        env_params=env_kwargs,
        env_class=env_class,
    )

    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False,
        env_params=env_kwargs,
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
