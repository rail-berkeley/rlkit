def experiment(variant):
    import os
    import os.path as osp

    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

    import torch
    import torch.nn as nn

    import rlkit.torch.pytorch_util as ptu
    from rlkit.core import logger
    from rlkit.envs.mujoco_vec_wrappers import DummyVecEnv, StableBaselinesVecEnv
    from rlkit.envs.multi_task_env import MultiTaskEnv
    from rlkit.envs.primitives_make_env import make_env
    from rlkit.torch.model_based.dreamer.actor_models import ActorModel
    from rlkit.torch.model_based.dreamer.dreamer_policy import (
        ActionSpaceSamplePolicy,
        DreamerLowLevelRAPSPolicy,
    )
    from rlkit.torch.model_based.dreamer.dreamer_v2 import DreamerV2LowLevelRAPSTrainer
    from rlkit.torch.model_based.dreamer.episode_replay_buffer import (
        EpisodeReplayBufferLowLevelRAPS,
    )
    from rlkit.torch.model_based.dreamer.kitchen_video_func import (
        post_epoch_visualize_func,
    )
    from rlkit.torch.model_based.dreamer.mlp import Mlp
    from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
    from rlkit.torch.model_based.dreamer.rollout_functions import (
        vec_rollout_low_level_raps,
    )
    from rlkit.torch.model_based.dreamer.train_world_model import (
        visualize_primitive_unsubsampled_rollout,
    )
    from rlkit.torch.model_based.dreamer.world_models import LowlevelRAPSWorldModel
    from rlkit.torch.model_based.rl_algorithm import TorchBatchRLAlgorithm

    env_suite = variant.get("env_suite", "kitchen")
    env_kwargs = variant["env_kwargs"]
    num_expl_envs = variant["num_expl_envs"]
    num_low_level_actions_per_primitive = variant["num_low_level_actions_per_primitive"]
    low_level_action_dim = variant["low_level_action_dim"]

    print("MAKING ENVS")
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
    expl_env.low_level_action_dim = low_level_action_dim
    eval_env.low_level_action_dim = low_level_action_dim
    eval_env.envs[0].low_level_action_dim = low_level_action_dim
    discrete_continuous_dist = variant["actor_kwargs"]["discrete_continuous_dist"]
    continuous_action_dim = eval_envs[0].max_arg_len
    discrete_action_dim = eval_envs[0].num_primitives
    if not discrete_continuous_dist:
        continuous_action_dim = continuous_action_dim + discrete_action_dim
        discrete_action_dim = 0
    action_dim = continuous_action_dim + discrete_action_dim
    obs_dim = expl_env.observation_space.low.size
    actor_model_class = ActorModel

    primitive_model = Mlp(
        hidden_sizes=variant["mlp_hidden_sizes"],
        output_size=variant["low_level_action_dim"],
        input_size=250 + eval_env.envs[0].action_space.low.shape[0] + 1,
        hidden_activation=nn.ReLU,
        apply_embedding=variant.get("primitive_embedding", False),
        num_embeddings=eval_envs[0].num_primitives,
        embedding_dim=eval_envs[0].num_primitives,
        embedding_slice=eval_envs[0].num_primitives,
    )
    world_model_class = LowlevelRAPSWorldModel
    world_model = world_model_class(
        low_level_action_dim,
        image_shape=eval_envs[0].image_shape,
        primitive_model=primitive_model,
        **variant["model_kwargs"],
    )
    actor = actor_model_class(
        variant["model_kwargs"]["model_hidden_size"],
        world_model.feature_size,
        hidden_activation=nn.ELU,
        discrete_action_dim=discrete_action_dim,
        continuous_action_dim=continuous_action_dim,
        **variant["actor_kwargs"],
    )
    vf = Mlp(
        hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]]
        * variant["vf_kwargs"]["num_layers"],
        output_size=1,
        input_size=world_model.feature_size,
        hidden_activation=nn.ELU,
    )
    target_vf = Mlp(
        hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]]
        * variant["vf_kwargs"]["num_layers"],
        output_size=1,
        input_size=world_model.feature_size,
        hidden_activation=nn.ELU,
    )

    if variant.get("models_path", None) is not None:
        filename = variant["models_path"]
        actor.load_state_dict(torch.load(osp.join(filename, "actor.ptc")))
        vf.load_state_dict(torch.load(osp.join(filename, "vf.ptc")))
        target_vf.load_state_dict(torch.load(osp.join(filename, "target_vf.ptc")))
        world_model.load_state_dict(torch.load(osp.join(filename, "world_model.ptc")))
        print("LOADED MODELS")

    expl_policy = DreamerLowLevelRAPSPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
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
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
        low_level_action_dim=low_level_action_dim,
        exploration=False,
        expl_amount=0.0,
        discrete_action_dim=discrete_action_dim,
        continuous_action_dim=continuous_action_dim,
        discrete_continuous_dist=discrete_continuous_dist,
    )
    expl_policy.num_primitives = eval_env.envs[0].num_primitives
    eval_policy.num_primitives = eval_env.envs[0].num_primitives

    rand_policy = ActionSpaceSamplePolicy(expl_env)
    rand_policy.num_primitives = eval_env.envs[0].num_primitives
    eval_env.num_low_level_actions_per_primitive = num_low_level_actions_per_primitive
    expl_env.num_low_level_actions_per_primitive = num_low_level_actions_per_primitive
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
        prioritize_fraction=variant["prioritize_fraction"],
        uniform_priorities=variant.get("uniform_priorities", True),
    )
    filename = variant.get("replay_buffer_path", None)
    if filename is not None:
        replay_buffer.load_buffer(filename, eval_env.envs[0].num_primitives)
    eval_filename = variant.get("eval_buffer_path", None)
    if eval_filename is not None:
        eval_buffer = EpisodeReplayBufferLowLevelRAPS(
            1000,
            expl_env,
            variant["algorithm_kwargs"]["max_path_length"],
            num_low_level_actions_per_primitive,
            obs_dim,
            action_dim,
            low_level_action_dim,
            replace=False,
        )
        eval_buffer.load_buffer(eval_filename, eval_env.envs[0].num_primitives)
    else:
        eval_buffer = None

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
        eval_buffer=eval_buffer,
    )
    algorithm.low_level_primitives = True
    if variant.get("generate_video", False):
        post_epoch_visualize_func(algorithm, 0)
    elif variant.get("unsubsampled_rollout", False):
        visualize_primitive_unsubsampled_rollout(
            make_env_lambda(),
            make_env_lambda(),
            make_env_lambda(),
            logger.get_snapshot_dir(),
            algorithm.max_path_length,
            num_low_level_actions_per_primitive,
            policy=eval_policy,
            img_size=64,
            num_rollouts=4,
        )
    else:
        if variant.get("save_video", False):
            algorithm.post_epoch_funcs.append(post_epoch_visualize_func)
        print("TRAINING")
        algorithm.to(ptu.device)
        algorithm.train()
        if variant.get("save_video", False):
            post_epoch_visualize_func(algorithm, -1)
