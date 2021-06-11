def experiment(variant):
    import os

    from rlkit.torch.model_based.plan2explore.mcts_policy import HybridMCTSPolicy
    from rlkit.torch.model_based.plan2explore.plan2explore_advanced_mcts import (
        Plan2ExploreAdvancedMCTSTrainer,
    )
    from rlkit.torch.model_based.plan2explore.plan2explore_mcts import (
        Plan2ExploreMCTSTrainer,
    )

    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    import torch
    from d4rl.kitchen.kitchen_envs import (
        KitchenBottomLeftBurnerV0,
        KitchenHingeCabinetV0,
        KitchenKettleV0,
        KitchenLightSwitchV0,
        KitchenMicrowaveV0,
        KitchenMultitaskAllV0,
        KitchenSlideCabinetV0,
        KitchenTopLeftBurnerV0,
    )

    import rlkit.torch.pytorch_util as ptu
    from rlkit.envs.mujoco_vec_wrappers import (
        DummyVecEnv,
        StableBaselinesVecEnv,
        make_env,
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
    from rlkit.torch.model_based.plan2explore.actor_models import (
        ConditionalContinuousActorModel,
    )
    from rlkit.torch.model_based.plan2explore.latent_space_models import (
        OneStepEnsembleModel,
    )
    from rlkit.torch.model_based.plan2explore.mcts_policy import (
        HybridAdvancedMCTSPolicy,
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
            variant["model_kwargs"]["embedding_size"] += 9
    else:
        world_model_class = WorldModel
    one_step_ensemble = OneStepEnsembleModel(
        action_dim=action_dim,
        embedding_size=variant["model_kwargs"]["embedding_size"],
        deterministic_state_size=variant["model_kwargs"]["deterministic_state_size"],
        hidden_size=variant["one_step_ensemble_kwargs"]["hidden_size"],
        num_layers=variant["one_step_ensemble_kwargs"]["num_layers"],
        num_models=variant["one_step_ensemble_kwargs"]["num_models"],
        output_embeddings=variant["one_step_ensemble_kwargs"]["output_embeddings"],
    )
    world_model = world_model_class(
        action_dim,
        image_shape=eval_envs[0].image_shape,
        **variant["model_kwargs"],
        env=eval_envs[0],
    )
    actor = ConditionalContinuousActorModel(
        [variant["model_kwargs"]["model_hidden_size"]] * 4,
        world_model.feature_size,
        hidden_activation=torch.nn.functional.elu,
        discrete_action_dim=num_primitives,
        continuous_action_dim=continuous_action_dim,
        use_tanh_normal=variant["actor_kwargs"]["use_tanh_normal"],
        mean_scale=variant["actor_kwargs"]["mean_scale"],
        init_std=variant["actor_kwargs"]["init_std"],
        env=eval_envs[0],
        use_per_primitive_actor=variant["actor_kwargs"]["use_per_primitive_actor"],
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

    exploration_actor = ConditionalContinuousActorModel(
        [variant["model_kwargs"]["model_hidden_size"]] * 4,
        variant["model_kwargs"]["stochastic_state_size"]
        + variant["model_kwargs"]["deterministic_state_size"],
        hidden_activation=torch.nn.functional.elu,
        discrete_action_dim=num_primitives,
        continuous_action_dim=continuous_action_dim,
        env=eval_envs[0],
        use_per_primitive_actor=variant["actor_kwargs"]["use_per_primitive_actor"],
    )
    exploration_vf = Mlp(
        hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]] * 3,
        output_size=1,
        input_size=variant["model_kwargs"]["stochastic_state_size"]
        + variant["model_kwargs"]["deterministic_state_size"],
        hidden_activation=torch.nn.functional.elu,
    )
    exploration_target_vf = Mlp(
        hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]] * 3,
        output_size=1,
        input_size=variant["model_kwargs"]["stochastic_state_size"]
        + variant["model_kwargs"]["deterministic_state_size"],
        hidden_activation=torch.nn.functional.elu,
    )
    variant["trainer_kwargs"]["exploration_target_vf"] = exploration_target_vf

    if variant.get("use_mcts_policy", False):

        # expl_policy = HybridMCTSPolicy(
        #     world_model,
        #     eval_envs[0].max_steps,
        #     eval_envs[0].num_primitives,
        #     eval_envs[0].action_space.low.size,
        #     eval_envs[0].action_space,
        #     exploration_actor,
        #     one_step_ensemble,
        #     **variant["expl_policy_kwargs"],
        # )
        # eval_policy = HybridMCTSPolicy(
        #     world_model,
        #     eval_envs[0].max_steps,
        #     eval_envs[0].num_primitives,
        #     eval_envs[0].action_space.low.size,
        #     eval_envs[0].action_space,
        #     eval_actor,
        #     one_step_ensemble,
        #     **variant["eval_policy_kwargs"],
        # )
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
    if variant.get("trainer_class") == "plan2explore_advanced_mcts":
        trainer_class = Plan2ExploreAdvancedMCTSTrainer
    else:
        trainer_class = Plan2ExploreMCTSTrainer
    trainer = trainer_class(
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
    algorithm.post_epoch_funcs.append(video_post_epoch_func)
    algorithm.to(ptu.device)
    algorithm.train()
    video_post_epoch_func(algorithm, -1)
