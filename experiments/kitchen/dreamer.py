from rlkit.launchers.launcher_util import run_experiment
import rlkit.util.hyperparameter as hyp
import argparse
import libtmux


def experiment(variant):

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
    from hrl_exp.envs.mujoco_vec_wrappers import Async, make_env, VecEnv
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

    expl_env = VecEnv(expl_envs)
    eval_env = VecEnv(expl_envs)
    max_path_length = expl_envs[0].max_steps
    variant["algorithm_kwargs"]["max_path_length"] = max_path_length
    variant["trainer_kwargs"]["imagination_horizon"] = max_path_length + 1

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


parser = argparse.ArgumentParser()
parser.add_argument("--exp_prefix", type=str, default="test")
parser.add_argument("--num_seeds", type=int, default=1)
parser.add_argument("--mode", type=str, default="local")
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--tmux", action="store_true", default=False)
parser.add_argument("--tmux_session_name", type=str, default="")
parser.add_argument("--num_expl_envs", type=int, default=10)
args = parser.parse_args()

if args.tmux:
    server = libtmux.Server()
    session = server.find_where({"session_name": args.tmux_session_name})
if args.debug:
    algorithm_kwargs = dict(
        num_epochs=2,
        num_eval_steps_per_epoch=30,
        num_trains_per_train_loop=10,
        num_expl_steps_per_train_loop=150,  # 200 samples since num_envs = 50 and max_path_length + 1 = 4
        min_num_steps_before_training=100,
        num_pretrain_steps=100,
        num_train_loops_per_epoch=1,
        max_path_length=3,
        batch_size=50,
    )
    exp_prefix = "test" + args.exp_prefix
else:
    algorithm_kwargs = dict(
        num_epochs=25,
        num_eval_steps_per_epoch=30,
        num_trains_per_train_loop=200,
        num_expl_steps_per_train_loop=150,  # 200 samples since num_envs = 50 and max_path_length + 1 = 4
        min_num_steps_before_training=5000,
        num_pretrain_steps=100,
        num_train_loops_per_epoch=5,
        max_path_length=3,
        batch_size=625,
    )
    exp_prefix = args.exp_prefix
variant = dict(
    algorithm="Dreamer",
    version="normal",
    replay_buffer_size=int(1e5),
    algorithm_kwargs=algorithm_kwargs,
    env_class="microwave",
    env_kwargs=dict(
        dense=False,
        delta=0.0,
        image_obs=True,
    ),
    model_kwargs=dict(
        model_hidden_size=400,
        stochastic_state_size=60,
        deterministic_state_size=400,
    ),
    trainer_kwargs=dict(
        discount=0.99,
        reward_scale=1.0,
        actor_lr=8e-5,
        vf_lr=8e-5,
        world_model_lr=6e-4,
        use_amp=False,
        opt_level="O1",
        gradient_clip=100.0,
        lam=0.95,
        imagination_horizon=algorithm_kwargs["max_path_length"] + 1,
        free_nats=3.0,
        kl_scale=1.0,
        optimizer_class="torch_adam",
        pcont_scale=10.0,
        use_pcont=True,
    ),
    num_expl_envs=args.num_expl_envs,
    num_eval_envs=1,
)

search_space = {
    "env_class": [
        #"microwave",
        #"kettle",
        #"top_burner",
        #"slide_cabinet",
        #"hinge_cabinet",
        "light_switch",
    ],
    "env_kwargs.delta": [
0.05, 
0.1,
0.15,
],
}
sweeper = hyp.DeterministicHyperparameterSweeper(
    search_space,
    default_parameters=variant,
)

for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
    for _ in range(args.num_seeds):
        run_experiment(
            experiment,
            exp_prefix=args.exp_prefix,
            mode="here_no_doodad",
            variant=variant,
            use_gpu=True,
            snapshot_mode="last",
        )
