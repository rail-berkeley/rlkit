def experiment(variant):
    from rlkit.core import logger

    if variant["algorithm_kwargs"]["use_wandb"]:
        import wandb

        with wandb.init(
            project=variant["exp_prefix"], name=variant["exp_name"], config=variant
        ):
            run_experiment(variant)
    else:
        run_experiment(variant)


def run_experiment(variant):
    import os

    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

    import torch
    from d4rl.kitchen.kitchen_envs import (
        KitchenHingeCabinetV0,
        KitchenKettleV0,
        KitchenLightSwitchV0,
        KitchenMicrowaveV0,
        KitchenMultitaskAllV0,
        KitchenSlideCabinetV0,
        KitchenTopLeftBurnerV0,
    )
    from hrl_exp.envs.mujoco_vec_wrappers import (
        DummyVecEnv,
        StableBaselinesVecEnv,
        make_env,
    )

    import rlkit.torch.pytorch_util as ptu
    from rlkit.torch.model_based.dreamer.actor_models import ActorModel
    from rlkit.torch.model_based.dreamer.dreamer import DreamerTrainer
    from rlkit.torch.model_based.dreamer.dreamer_policy import (
        ActionSpaceSamplePolicy,
        DreamerPolicy,
    )
    from rlkit.torch.model_based.dreamer.dreamer_v2 import DreamerV2Trainer
    from rlkit.torch.model_based.dreamer.episode_replay_buffer import (
        EpisodeReplayBuffer,
    )
    from rlkit.torch.model_based.dreamer.kitchen_video_func import video_post_epoch_func
    from rlkit.torch.model_based.dreamer.mlp import Mlp
    from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
    from rlkit.torch.model_based.dreamer.world_models import (
        MultitaskWorldModel,
        WorldModel,
    )
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

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
    elif env_class == "top_left_burner":
        env_class_ = KitchenTopLeftBurnerV0
    elif env_class == "light_switch":
        env_class_ = KitchenLightSwitchV0
    elif env_class == "multitask_all":
        env_class_ = KitchenMultitaskAllV0
    else:
        raise EnvironmentError("invalid env provided")

    env_fns = [
        lambda: make_env(
            env_class=env_class_,
            env_kwargs=variant["env_kwargs"],
        )
        for _ in range(variant["num_expl_envs"])
    ]
    expl_env = StableBaselinesVecEnv(env_fns=env_fns, start_method="fork")

    eval_envs = [
        make_env(
            env_class=env_class_,
            env_kwargs=variant["env_kwargs"],
        )
    ]

    eval_env = DummyVecEnv(eval_envs)
    max_path_length = eval_envs[0].max_steps
    variant["algorithm_kwargs"]["max_path_length"] = max_path_length
    variant["trainer_kwargs"]["imagination_horizon"] = max_path_length + 1

    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    num_primitives = eval_envs[0].num_primitives
    max_arg_len = eval_envs[0].max_arg_len

    if variant.get("world_model_class", "world_model") == "multitask":
        world_model_class = MultitaskWorldModel
    else:
        world_model_class = WorldModel
    world_model = world_model_class(
        action_dim,
        **variant["model_kwargs"],
    )

    if (
        variant["actor_kwargs"]["discrete_continuous_dist"]
        or variant["env_kwargs"]["fixed_schema"]
    ):
        continuous_action_dim = max_arg_len
    else:
        continuous_action_dim = max_arg_len + num_primitives

    if variant.get("algorithm", "dreamer") == "dreamer_v2":
        trainer_class = DreamerV2Trainer
    else:
        trainer_class = DreamerTrainer

    actor = ActorModel(
        [variant["model_kwargs"]["model_hidden_size"]] * 4,
        world_model.feature_size,
        hidden_activation=torch.nn.functional.elu,
        discrete_action_dim=num_primitives,
        continuous_action_dim=continuous_action_dim,
        discrete_continuous_dist=variant["actor_kwargs"]["discrete_continuous_dist"]
        and (not variant["env_kwargs"]["fixed_schema"]),
    )
    vf = Mlp(
        hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]]
        * variant["vf_kwargs"]["num_layers"],
        output_size=1,
        input_size=world_model.feature_size,
        hidden_activation=torch.nn.functional.elu,
    )
    target_vf = Mlp(
        hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]]
        * variant["vf_kwargs"]["num_layers"],
        output_size=1,
        input_size=world_model.feature_size,
        hidden_activation=torch.nn.functional.elu,
    )
    variant["trainer_kwargs"]["target_vf"] = target_vf

    expl_policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        exploration=True,
        expl_amount=variant.get("expl_amount", 0.3),
        discrete_action_dim=num_primitives,
        continuous_action_dim=continuous_action_dim,
        discrete_continuous_dist=variant["actor_kwargs"]["discrete_continuous_dist"]
        and (not variant["env_kwargs"]["fixed_schema"]),
    )
    eval_policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        exploration=False,
        expl_amount=0.0,
        discrete_action_dim=num_primitives,
        continuous_action_dim=continuous_action_dim,
        discrete_continuous_dist=variant["actor_kwargs"]["discrete_continuous_dist"]
        and (not variant["env_kwargs"]["fixed_schema"]),
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
    trainer = trainer_class(
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
    # algorithm.post_epoch_funcs.append(video_post_epoch_func)
    algorithm.to(ptu.device)
    algorithm.train()
    # video_post_epoch_func(algorithm, -1)
