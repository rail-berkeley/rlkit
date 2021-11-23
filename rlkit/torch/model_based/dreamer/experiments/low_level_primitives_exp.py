def experiment(variant):
    import os

    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

    import torch

    import rlkit.torch.pytorch_util as ptu
    from rlkit.envs.mujoco_vec_wrappers import DummyVecEnv, StableBaselinesVecEnv
    from rlkit.envs.multi_task_env import MultiTaskEnv
    from rlkit.envs.primitives_make_env import make_env
    from rlkit.torch.model_based.dreamer.actor_models import (
        ActorModel,
        ConditionalActorModel,
    )
    from rlkit.torch.model_based.dreamer.dreamer import DreamerTrainer
    from rlkit.torch.model_based.dreamer.dreamer_policy import (
        ActionSpaceSamplePolicy,
        DreamerLowLevelRAPSPolicy,
        DreamerPolicy,
    )
    from rlkit.torch.model_based.dreamer.dreamer_v2 import (
        DreamerV2LowLevelRAPSTrainer,
        DreamerV2Trainer,
    )
    from rlkit.torch.model_based.dreamer.episode_replay_buffer import (
        EpisodeReplayBuffer,
        EpisodeReplayBufferLowLevelRAPS,
    )
    from rlkit.torch.model_based.dreamer.kitchen_video_func import video_post_epoch_func
    from rlkit.torch.model_based.dreamer.mcts.dreamer_v2_mcts import (
        DreamerV2MCTSTrainer,
    )
    from rlkit.torch.model_based.dreamer.mlp import Mlp
    from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
    from rlkit.torch.model_based.dreamer.rollout_functions import (
        vec_rollout_low_level_raps,
    )
    from rlkit.torch.model_based.dreamer.world_models import (
        LowlevelRAPSWorldModel,
        StateConcatObsWorldModel,
        WorldModel,
    )
    from rlkit.torch.model_based.plan2explore.actor_models import (
        ConditionalContinuousActorModel,
    )
    from rlkit.torch.model_based.plan2explore.mcts.mcts_policy import (
        HybridAdvancedMCTSPolicy,
    )
    from rlkit.torch.model_based.rl_algorithm import TorchBatchRLAlgorithm

    env_suite = variant.get("env_suite", "kitchen")
    env_kwargs = variant["env_kwargs"]
    use_raw_actions = variant["use_raw_actions"]
    num_expl_envs = variant["num_expl_envs"]
    actor_model_class_name = variant.get("actor_model_class", "actor_model")
    num_low_level_actions_per_primitive = variant["num_low_level_actions_per_primitive"]
    low_level_action_dim = variant["low_level_action_dim"]
    if variant.get("make_multi_task_env", False):
        make_env_lambda = lambda: MultiTaskEnv(
            env_suite, variant["env_names"], env_kwargs
        )
        # TODO: make the make env lambda test have different environments
    else:
        env_name = variant["env_name"]
        make_env_lambda = lambda: make_env(env_suite, env_name, env_kwargs)

    if num_expl_envs > 1:
        env_fns = [make_env_lambda for _ in range(num_expl_envs)]
        expl_env = StableBaselinesVecEnv(env_fns=env_fns, start_method="fork")
    else:
        expl_envs = [make_env_lambda()]
        expl_env = DummyVecEnv(
            expl_envs, pass_render_kwargs=variant.get("pass_render_kwargs", False)
        )
    eval_envs = [make_env_lambda() for _ in range(1)]
    eval_env = DummyVecEnv(
        eval_envs, pass_render_kwargs=variant.get("pass_render_kwargs", False)
    )
    if use_raw_actions:
        discrete_continuous_dist = False
        continuous_action_dim = eval_env.action_space.low.size
        discrete_action_dim = 0
        action_dim = continuous_action_dim
    else:
        discrete_continuous_dist = variant["actor_kwargs"]["discrete_continuous_dist"]
        continuous_action_dim = eval_envs[0].max_arg_len
        discrete_action_dim = eval_envs[0].num_primitives
        if not discrete_continuous_dist:
            continuous_action_dim = continuous_action_dim + discrete_action_dim
            discrete_action_dim = 0
        action_dim = continuous_action_dim + discrete_action_dim
    obs_dim = expl_env.observation_space.low.size
    if actor_model_class_name == "conditional_actor_model":
        actor_model_class = ConditionalActorModel
    elif actor_model_class_name == "continuous_conditional_actor_model":
        actor_model_class = ConditionalContinuousActorModel
    elif actor_model_class_name == "actor_model":
        actor_model_class = ActorModel

    primitive_model = Mlp(
        hidden_sizes=variant["mlp_hidden_sizes"],
        output_size=variant["low_level_action_dim"],
        input_size=250 + eval_env.envs[0].action_space.low.shape[0] + 1,
        hidden_activation=torch.nn.functional.relu,
    ).to(ptu.device)
    if variant.get("load_from_path", False):
        filename = variant["models_path"] + variant["pkl_file_name"]
        print(filename)
        data = torch.load(filename)
        actor = data["trainer/actor"]
        vf = data["trainer/vf"]
        target_vf = data["trainer/target_vf"]
        world_model = data["trainer/world_model"]
    else:
        world_model_class = LowlevelRAPSWorldModel
        world_model = world_model_class(
            low_level_action_dim,
            image_shape=eval_envs[0].image_shape,
            primitive_model=primitive_model,
            **variant["model_kwargs"],
        )
    if variant.get("retrain_actor_and_vf", True):
        actor = actor_model_class(
            variant["model_kwargs"]["model_hidden_size"],
            world_model.feature_size,
            hidden_activation=torch.nn.functional.elu,
            discrete_action_dim=discrete_action_dim,
            continuous_action_dim=continuous_action_dim,
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

    if variant.get("use_mcts_policy", False):
        expl_policy = HybridAdvancedMCTSPolicy(
            world_model,
            discrete_action_dim,
            action_dim,
            eval_envs[0].action_space,
            actor,
            None,
            vf,
            **variant["expl_policy_kwargs"],
        )
        eval_policy = HybridAdvancedMCTSPolicy(
            world_model,
            discrete_action_dim,
            action_dim,
            eval_envs[0].action_space,
            actor,
            None,
            vf,
            **variant["eval_policy_kwargs"],
        )
    else:
        expl_policy = DreamerLowLevelRAPSPolicy(
            world_model,
            actor,
            obs_dim,
            action_dim,
            primitive_model=primitive_model,
            num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
            low_level_action_dim=low_level_action_dim,
            exploration=True,
            expl_amount=variant.get("expl_amount", 0.3),
            discrete_action_dim=discrete_action_dim,
            continuous_action_dim=continuous_action_dim,
            discrete_continuous_dist=discrete_continuous_dist,
        )
        eval_policy = DreamerLowLevelRAPSPolicy(
            world_model,
            actor,
            obs_dim,
            action_dim,
            primitive_model=primitive_model,
            num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
            low_level_action_dim=low_level_action_dim,
            exploration=False,
            expl_amount=0.0,
            discrete_action_dim=discrete_action_dim,
            continuous_action_dim=continuous_action_dim,
            discrete_continuous_dist=discrete_continuous_dist,
        )

    rand_policy = ActionSpaceSamplePolicy(expl_env)

    expl_path_collector = VecMdpPathCollector(
        expl_env,
        expl_policy,
        save_env_in_snapshot=False,
        rollout_fn=vec_rollout_low_level_raps,
    )

    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False,
        rollout_fn=vec_rollout_low_level_raps,
    )

    replay_buffer = EpisodeReplayBufferLowLevelRAPS(
        variant["replay_buffer_size"],
        expl_env,
        variant["algorithm_kwargs"]["max_path_length"],
        num_low_level_actions_per_primitive,
        obs_dim,
        action_dim,
        low_level_action_dim,
        replace=False,
    )
    trainer = DreamerV2LowLevelRAPSTrainer(
        eval_env,
        actor,
        vf,
        target_vf,
        world_model,
        eval_envs[0].image_shape,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
        num_primitives=eval_env.envs[0].num_primitives,
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
    if variant.get("save_video", False):
        algorithm.post_epoch_funcs.append(video_post_epoch_func)
    print("TRAINING")
    algorithm.to(ptu.device)
    algorithm.train()
    if variant.get("save_video", False):
        video_post_epoch_func(algorithm, -1)
