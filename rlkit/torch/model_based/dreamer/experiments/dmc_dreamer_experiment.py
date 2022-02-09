def experiment(variant):
    import os

    from rlkit.envs.wrappers.dmc_wrappers import (
        ActionRepeat,
        DeepMindControl,
        NormalizeActions,
        TimeLimit,
    )
    from rlkit.torch.model_based.dreamer.visualization import post_epoch_visualize_func

    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    import torch
    import torch.nn as nn

    import rlkit.torch.pytorch_util as ptu
    from rlkit.envs.wrappers.mujoco_vec_wrappers import DummyVecEnv
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
    from rlkit.torch.model_based.dreamer.mlp import Mlp
    from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
    from rlkit.torch.model_based.dreamer.world_models import WorldModel
    from rlkit.torch.model_based.rl_algorithm import TorchBatchRLAlgorithm

    expl_env = DeepMindControl(variant["env_id"])
    expl_env.reset()
    expl_env = ActionRepeat(expl_env, 2)
    expl_env = NormalizeActions(expl_env)
    expl_env = DummyVecEnv([TimeLimit(expl_env, 500)], pass_render_kwargs=False)

    eval_env = DeepMindControl(variant["env_id"])
    eval_env.reset()
    eval_env = ActionRepeat(eval_env, 2)
    eval_env = NormalizeActions(eval_env)
    eval_env = DummyVecEnv([TimeLimit(eval_env, 500)], pass_render_kwargs=False)

    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    world_model_class = WorldModel

    world_model = world_model_class(
        action_dim,
        image_shape=(3, 64, 64),
        **variant["model_kwargs"],
    )
    actor = ActorModel(
        variant["model_kwargs"]["model_hidden_size"],
        world_model.feature_size,
        hidden_activation=nn.ELU,
        discrete_action_dim=0,
        continuous_action_dim=eval_env.action_space.low.size,
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
    variant["trainer_kwargs"]["target_vf"] = target_vf

    expl_policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        exploration=True,
        expl_amount=variant.get("expl_amount", 0.3),
        discrete_action_dim=0,
        continuous_action_dim=eval_env.action_space.low.size,
    )
    eval_policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        exploration=False,
        expl_amount=0.0,
        discrete_action_dim=0,
        continuous_action_dim=eval_env.action_space.low.size,
    )

    rand_policy = ActionSpaceSamplePolicy(expl_env)

    expl_path_collector = VecMdpPathCollector(
        expl_env,
        expl_policy,
        save_env_in_snapshot=False,
        env_params={},
        env_class={},
    )

    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False,
        env_params={},
        env_class={},
    )

    replay_buffer = EpisodeReplayBuffer(
        1,
        obs_dim,
        action_dim,
        variant["replay_buffer_size"],
        500,
        replace=False,
        use_batch_length=True,
        batch_length=50,
    )
    trainer_class_name = variant.get("algorithm", "DreamerV2")
    if trainer_class_name == "DreamerV2":
        trainer_class = DreamerV2Trainer
    else:
        trainer_class = DreamerTrainer
    trainer = trainer_class(
        world_model=world_model,
        actor=actor,
        vf=vf,
        image_shape=(3, 64, 64),
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
    print("TRAINING")
    algorithm.to(ptu.device)
    algorithm.train()
