def experiment(variant):
    import os
    import os.path as osp

    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

    import torch
    import torch.nn as nn

    import rlkit.envs.primitives_make_env as primitives_make_env
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
    from rlkit.torch.model_based.dreamer.dreamer_v2 import DreamerV2Trainer
    from rlkit.torch.model_based.dreamer.episode_replay_buffer import (
        EpisodeReplayBuffer,
        EpisodeReplayBufferLowLevelRAPS,
    )
    from rlkit.torch.model_based.dreamer.mlp import Mlp
    from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
    from rlkit.torch.model_based.dreamer.visualization import post_epoch_visualize_func
    from rlkit.torch.model_based.dreamer.world_models import WorldModel
    from rlkit.torch.model_based.rl_algorithm import TorchBatchRLAlgorithm

    env_suite = variant.get("env_suite", "kitchen")
    env_name = variant["env_name"]
    env_kwargs = variant["env_kwargs"]
    use_raw_actions = variant["use_raw_actions"]
    num_expl_envs = variant["num_expl_envs"]
    if num_expl_envs > 1:
        env_fns = [
            lambda: primitives_make_env.make_env(env_suite, env_name, env_kwargs)
            for _ in range(num_expl_envs)
        ]
        expl_env = StableBaselinesVecEnv(
            env_fns=env_fns,
            start_method="fork",
            reload_state_args=(
                num_expl_envs,
                primitives_make_env.make_env,
                (env_suite, env_name, env_kwargs),
            ),
        )
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
    obs_dim = expl_env.observation_space.low.size

    world_model = WorldModel(
        action_dim,
        image_shape=eval_envs[0].image_shape,
        **variant["model_kwargs"],
    )
    actor = ActorModel(
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

    expl_policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        exploration=True,
        expl_amount=variant.get("expl_amount", 0.3),
        discrete_action_dim=discrete_action_dim,
        continuous_action_dim=continuous_action_dim,
        discrete_continuous_dist=discrete_continuous_dist,
    )
    eval_policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
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
    )

    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False,
    )

    variant["replay_buffer_kwargs"]["use_batch_length"] = use_batch_length
    replay_buffer = EpisodeReplayBuffer(
        num_expl_envs,
        obs_dim,
        action_dim,
        **variant["replay_buffer_kwargs"],
    )
    eval_filename = variant.get("eval_buffer_path", None)
    if eval_filename is not None:
        eval_buffer = EpisodeReplayBufferLowLevelRAPS(
            1000,
            expl_env,
            variant["algorithm_kwargs"]["max_path_length"],
            10,
            obs_dim,
            action_dim,
            9,
            replace=False,
        )
        eval_buffer.load_buffer(eval_filename, eval_env.envs[0].num_primitives)
    else:
        eval_buffer = None
    trainer = DreamerV2Trainer(
        actor,
        vf,
        target_vf,
        world_model,
        eval_envs[0].image_shape,
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
        eval_buffer=eval_buffer,
        **variant["algorithm_kwargs"],
    )
    algorithm.low_level_primitives = False
    if variant.get("generate_video", False):
        post_epoch_visualize_func(algorithm, 0)
    else:
        if variant.get("save_video", False):
            algorithm.post_epoch_funcs.append(post_epoch_visualize_func)
        print("TRAINING")
        algorithm.to(ptu.device)
        algorithm.train()
        if variant.get("save_video", False):
            post_epoch_visualize_func(algorithm, -1)
