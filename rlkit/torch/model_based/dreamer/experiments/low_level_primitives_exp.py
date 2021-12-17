import h5py
import numpy as np

from rlkit.torch.model_based.dreamer.kitchen_video_func import video_low_level_func


def experiment(variant):
    import gc
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
    from rlkit.torch.model_based.dreamer.dreamer_policy import (
        ActionSpaceSamplePolicy,
        DreamerLowLevelRAPSPolicy,
    )
    from rlkit.torch.model_based.dreamer.dreamer_v2 import DreamerV2LowLevelRAPSTrainer
    from rlkit.torch.model_based.dreamer.episode_replay_buffer import (
        EpisodeReplayBufferLowLevelRAPS,
    )
    from rlkit.torch.model_based.dreamer.mlp import Mlp
    from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
    from rlkit.torch.model_based.dreamer.rollout_functions import (
        vec_rollout_low_level_raps,
    )
    from rlkit.torch.model_based.dreamer.world_models import LowlevelRAPSWorldModel
    from rlkit.torch.model_based.plan2explore.actor_models import (
        ConditionalContinuousActorModel,
    )
    from rlkit.torch.model_based.rl_algorithm import TorchBatchRLAlgorithm

    env_suite = variant.get("env_suite", "kitchen")
    env_kwargs = variant["env_kwargs"]
    num_expl_envs = variant["num_expl_envs"]
    actor_model_class_name = variant.get("actor_model_class", "actor_model")
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
    )
    world_model_class = LowlevelRAPSWorldModel
    world_model = world_model_class(
        low_level_action_dim,
        image_shape=eval_envs[0].image_shape,
        primitive_model=primitive_model,
        **variant["model_kwargs"],
    )
    if variant.get("world_model_path", None) is not None:
        world_model.load_state_dict(torch.load(variant["world_model_path"]))

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
    )
    filename = variant.get("replay_buffer_path", None)
    if filename is not None:
        print("LOADING REPLAY BUFFER")
        with h5py.File(filename, "r") as f:
            observations = np.array(f["observations"][:])
            low_level_actions = np.array(f["low_level_actions"][:])
            high_level_actions = np.array(f["high_level_actions"][:])
            rewards = np.array(f["rewards"][:])
            terminals = np.array(f["terminals"][:])
        num_trajs = observations.shape[0]
        replay_buffer._observations[:num_trajs] = observations
        replay_buffer._low_level_actions[:num_trajs] = low_level_actions
        argmax = np.argmax(
            high_level_actions[:, :, : eval_env.envs[0].num_primitives], axis=-1
        )
        one_hots = np.eye(eval_env.envs[0].num_primitives)[argmax]
        one_hots[:, 0:1, :] = np.zeros(
            (one_hots.shape[0], 1, eval_env.envs[0].num_primitives)
        )
        high_level_actions = np.concatenate(
            (one_hots, high_level_actions[:, :, eval_env.envs[0].num_primitives :]),
            axis=-1,
        )
        replay_buffer._high_level_actions[:num_trajs] = high_level_actions
        replay_buffer._rewards[:num_trajs] = rewards
        replay_buffer._terminals[:num_trajs] = terminals
        replay_buffer._top = num_trajs
        replay_buffer._size = num_trajs

        del observations
        del low_level_actions
        del high_level_actions
        del rewards
        del terminals
        gc.collect()

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
    print("NODENAME: ", os.environ["SLURMD_NODENAME"])
    print()
    if variant.get("save_video", False):
        algorithm.post_epoch_funcs.append(video_low_level_func)
    print("TRAINING")
    algorithm.to(ptu.device)
    algorithm.train()
    if variant.get("save_video", False):
        video_low_level_func(algorithm, -1)
