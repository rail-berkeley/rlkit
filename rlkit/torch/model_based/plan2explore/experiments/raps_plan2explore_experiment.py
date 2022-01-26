def experiment(variant):
    import os

    import rlkit.envs.primitives_make_env as primitives_make_env

    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    import torch

    import rlkit.torch.pytorch_util as ptu
    from rlkit.envs.wrappers.mujoco_vec_wrappers import (
        DummyVecEnv,
        StableBaselinesVecEnv,
    )
    from rlkit.torch.model_based.dreamer.actor_models import ActorModel
    from rlkit.torch.model_based.dreamer.dreamer_policy import (
        ActionSpaceSamplePolicy,
        DreamerPolicy,
    )
    from rlkit.torch.model_based.dreamer.episode_replay_buffer import (
        EpisodeReplayBuffer,
    )
    from rlkit.torch.model_based.dreamer.mlp import Mlp
    from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
    from rlkit.torch.model_based.dreamer.visualization import video_post_epoch_func
    from rlkit.torch.model_based.dreamer.world_models import WorldModel
    from rlkit.torch.model_based.plan2explore.latent_space_models import (
        OneStepEnsembleModel,
    )
    from rlkit.torch.model_based.plan2explore.plan2explore import Plan2ExploreTrainer
    from rlkit.torch.model_based.rl_algorithm import TorchBatchRLAlgorithm

    env_suite = variant.get("env_suite", "kitchen")
    env_name = variant["env_name"]
    env_kwargs = variant["env_kwargs"]
    use_raw_actions = variant["use_raw_actions"]
    num_expl_envs = variant["num_expl_envs"]
    actor_model_class_name = variant.get("actor_model_class", "actor_model")

    if num_expl_envs > 1:
        env_fns = [
            lambda: primitives_make_env.make_env(env_suite, env_name, env_kwargs)
            for _ in range(num_expl_envs)
        ]
        expl_env = StableBaselinesVecEnv(env_fns=env_fns, start_method="fork")
    else:
        expl_envs = [primitives_make_env.make_env(env_suite, env_name, env_kwargs)]
        expl_env = DummyVecEnv(
            expl_envs, pass_render_kwargs=variant.get("pass_render_kwargs", False)
        )
    eval_envs = [
        primitives_make_env.make_env(env_suite, env_name, env_kwargs) for _ in range(1)
    ]
    eval_env = DummyVecEnv(
        eval_envs, pass_render_kwargs=variant.get("pass_render_kwargs", False)
    )
    if use_raw_actions:
        discrete_continuous_dist = False
        continuous_action_dim = eval_env.action_space.low.size
        discrete_action_dim = 0
        use_batch_length = True
        action_dim = continuous_action_dim
    else:
        discrete_continuous_dist = variant["actor_kwargs"]["discrete_continuous_dist"]
        continuous_action_dim = eval_envs[0].max_arg_len
        discrete_action_dim = eval_envs[0].num_primitives
        if not discrete_continuous_dist:
            continuous_action_dim = continuous_action_dim + discrete_action_dim
            discrete_action_dim = 0
        action_dim = continuous_action_dim + discrete_action_dim
        use_batch_length = False
    world_model_class = WorldModel
    obs_dim = expl_env.observation_space.low.size
    actor_model_class = ActorModel
    if variant.get("load_from_path", False):
        data = torch.load(variant["models_path"])
        actor = data["trainer/actor"]
        vf = data["trainer/vf"]
        target_vf = data["trainer/target_vf"]
        world_model = data["trainer/world_model"]
    else:
        world_model = world_model_class(
            action_dim,
            image_shape=eval_envs[0].image_shape,
            **variant["model_kwargs"],
            env=eval_envs[0].env,
        )
        actor = actor_model_class(
            variant["model_kwargs"]["model_hidden_size"],
            world_model.feature_size,
            hidden_activation=torch.nn.functional.elu,
            discrete_action_dim=discrete_action_dim,
            continuous_action_dim=continuous_action_dim,
            env=eval_envs[0].env,
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
        discrete_action_dim=discrete_action_dim,
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

    if variant.get("expl_with_exploration_actor", True):
        expl_actor = exploration_actor
    else:
        expl_actor = actor
    expl_policy = DreamerPolicy(
        world_model,
        expl_actor,
        obs_dim,
        action_dim,
        exploration=True,
        expl_amount=variant.get("expl_amount", 0.3),
        discrete_action_dim=discrete_action_dim,
        continuous_action_dim=continuous_action_dim,
        discrete_continuous_dist=variant["actor_kwargs"]["discrete_continuous_dist"],
    )
    if variant.get("eval_with_exploration_actor", False):
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
        discrete_action_dim=discrete_action_dim,
        continuous_action_dim=continuous_action_dim,
        discrete_continuous_dist=variant["actor_kwargs"]["discrete_continuous_dist"],
    )

    rand_policy = ActionSpaceSamplePolicy(expl_env)

    expl_path_collector = VecMdpPathCollector(
        expl_env,
        expl_policy,
        save_env_in_snapshot=False,
    )

    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False,
    )

    replay_buffer = EpisodeReplayBuffer(
        variant["replay_buffer_size"],
        expl_env,
        variant["algorithm_kwargs"]["max_path_length"] + 1,
        obs_dim,
        action_dim,
        replace=False,
        use_batch_length=use_batch_length,
    )
    trainer = Plan2ExploreTrainer(
        eval_env,
        actor,
        vf,
        target_vf,
        world_model,
        eval_envs[0].image_shape,
        exploration_actor,
        exploration_vf,
        exploration_target_vf,
        one_step_ensemble,
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
