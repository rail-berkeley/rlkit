def experiment(variant):
    import os

    from rlkit.torch.model_based.plan2explore.mcts_policy import (
        HybridAdvancedMCTSPolicy,
    )

    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    import torch
    from d4rl.kitchen.kitchen_envs import (
        KitchenBottomLeftBurnerV0,
        KitchenHingeCabinetV0,
        KitchenKettleV0,
        KitchenLightSwitchV0,
        KitchenMicrowaveV0,
        KitchenSlideCabinetV0,
        KitchenTopLeftBurnerV0,
    )
    from hrl_exp.envs.mujoco_vec_wrappers import (
        DummyVecEnv,
        StableBaselinesVecEnv,
        make_env,
    )

    import rlkit.torch.pytorch_util as ptu
    from rlkit.torch.model_based.dreamer.actor_models import (
        ActorModel,
        ConditionalActorModel,
    )
    from rlkit.torch.model_based.dreamer.dreamer_policy import (
        ActionSpaceSamplePolicy,
        DreamerPolicy,
    )
    from rlkit.torch.model_based.dreamer.episode_replay_buffer import (
        EpisodeReplayBuffer,
    )
    from rlkit.torch.model_based.dreamer.kitchen_video_func import video_post_epoch_func
    from rlkit.torch.model_based.dreamer.mlp import Mlp
    from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
    from rlkit.torch.model_based.dreamer.world_models import (
        StateConcatObsWorldModel,
        WorldModel,
    )
    from rlkit.torch.model_based.plan2explore.latent_space_models import (
        OneStepEnsembleModel,
    )
    from rlkit.torch.model_based.plan2explore.plan2explore import Plan2ExploreTrainer
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
    elif env_class == "bottom_left_burner":
        env_class_ = KitchenBottomLeftBurnerV0
    elif env_class == "light_switch":
        env_class_ = KitchenLightSwitchV0
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

    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    num_primitives = eval_envs[0].num_primitives
    max_arg_len = eval_envs[0].max_arg_len

    if (
        variant["actor_kwargs"]["discrete_continuous_dist"]
        or variant["env_kwargs"]["fixed_schema"]
    ):
        continuous_action_dim = max_arg_len
    else:
        continuous_action_dim = max_arg_len + num_primitives
    if (
        variant.get("world_model_class", "world_model") == "multitask"
        or eval_envs[0].proprioception
    ):
        world_model_class = StateConcatObsWorldModel
        if eval_envs[0].proprioception:
            variant["model_kwargs"]["embedding_size"] += variant["model_kwargs"][
                "state_output_size"
            ]
    else:
        world_model_class = WorldModel

    world_model = world_model_class(
        action_dim,
        image_shape=eval_envs[0].image_shape,
        **variant["model_kwargs"],
        env=eval_envs[0],
    )
    if variant.get("actor_model_class", "actor_model") == "conditional_actor_model":
        actor_model_class = ConditionalActorModel
    else:
        actor_model_class = ActorModel
    actor = actor_model_class(
        variant["model_kwargs"]["model_hidden_size"],
        world_model.feature_size,
        hidden_activation=torch.nn.functional.elu,
        discrete_action_dim=num_primitives,
        continuous_action_dim=continuous_action_dim,
        env=eval_envs[0],
        **variant["actor_kwargs"],
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

    one_step_ensemble = OneStepEnsembleModel(
        action_dim=action_dim,
        embedding_size=variant["model_kwargs"]["embedding_size"],
        deterministic_state_size=variant["model_kwargs"]["deterministic_state_size"],
        stochastic_state_size=variant["model_kwargs"]["stochastic_state_size"],
        **variant["one_step_ensemble_kwargs"],
    )

    exploration_actor = actor_model_class(
        variant["model_kwargs"]["model_hidden_size"],
        world_model.feature_size,
        hidden_activation=torch.nn.functional.elu,
        discrete_action_dim=num_primitives,
        continuous_action_dim=continuous_action_dim,
        env=eval_envs[0],
        **variant["actor_kwargs"],
    )
    exploration_vf = Mlp(
        hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]]
        * variant["vf_kwargs"]["num_layers"],
        output_size=1,
        input_size=world_model.feature_size,
        hidden_activation=torch.nn.functional.elu,
    )
    exploration_target_vf = Mlp(
        hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]]
        * variant["vf_kwargs"]["num_layers"],
        output_size=1,
        input_size=world_model.feature_size,
        hidden_activation=torch.nn.functional.elu,
    )
    variant["trainer_kwargs"]["exploration_target_vf"] = exploration_target_vf

    if variant.get("use_mcts_policy", False):
        expl_policy = HybridAdvancedMCTSPolicy(
            world_model,
            eval_envs[0].max_steps,
            eval_envs[0].num_primitives,
            eval_envs[0].action_space.low.size,
            eval_envs[0].action_space,
            exploration_actor,
            one_step_ensemble,
            exploration_vf,
            **variant["expl_policy_kwargs"],
        )
        eval_policy = HybridAdvancedMCTSPolicy(
            world_model,
            eval_envs[0].max_steps,
            eval_envs[0].num_primitives,
            eval_envs[0].action_space.low.size,
            eval_envs[0].action_space,
            actor,
            one_step_ensemble,
            vf,
            **variant["eval_policy_kwargs"],
        )
    else:
        expl_policy = DreamerPolicy(
            world_model,
            exploration_actor,
            obs_dim,
            action_dim,
            exploration=True,
            expl_amount=variant.get("expl_amount", 0.3),
            discrete_action_dim=num_primitives,
            continuous_action_dim=continuous_action_dim,
            discrete_continuous_dist=variant["actor_kwargs"]["discrete_continuous_dist"]
            and (not variant["env_kwargs"]["fixed_schema"]),
        )
        if variant["eval_with_exploration_actor"]:
            eval_actor = exploration_actor
        else:
            eval_actor = actor
        eval_policy = DreamerPolicy(
            world_model,
            eval_actor,
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
        variant["algorithm_kwargs"]["max_path_length"] + 1,
        obs_dim,
        action_dim,
        replace=False,
    )
    trainer = Plan2ExploreTrainer(
        env=eval_env,
        world_model=world_model,
        actor=actor,
        vf=vf,
        image_shape=eval_envs[0].image_shape,
        one_step_ensemble=one_step_ensemble,
        exploration_actor=exploration_actor,
        exploration_vf=exploration_vf,
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
