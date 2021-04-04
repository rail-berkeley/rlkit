import gym
import numpy as np


def experiment(variant):
    from rlkit.core import logger

    # if variant["algorithm_kwargs"]["use_wandb"]:
    #     import wandb
    #     with wandb.init(
    #         project=variant["exp_prefix"], name=variant["exp_name"], config=variant
    #     ):
    #         run_experiment(variant)
    # else:
    run_experiment(variant)


def run_experiment(variant):
    import os

    import gym

    gym.logger.set_level(40)

    from rlkit.envs.dmc_wrappers import (
        ActionRepeat,
        ImageEnvMetaworld,
        NormalizeActions,
        TimeLimit,
    )

    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

    import torch
    from d4rl.kitchen.kitchen_envs import (
        KitchenHingeCabinetV0,
        KitchenHingeSlideBottomLeftBurnerLightV0,
        KitchenKettleV0,
        KitchenLightSwitchV0,
        KitchenMicrowaveKettleLightTopLeftBurnerV0,
        KitchenMicrowaveV0,
        KitchenSlideCabinetV0,
        KitchenTopLeftBurnerV0,
    )
    from hrl_exp.envs.mujoco_vec_wrappers import (
        DummyVecEnv,
        StableBaselinesVecEnv,
        make_env,
        make_env_multiworld,
    )

    import rlkit.torch.pytorch_util as ptu
    from rlkit.torch.model_based.dreamer.actor_models import (
        ActorModel,
        ConditionalActorModel,
    )
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
        StateConcatObsWorldModel,
        WorldModel,
    )
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

    env_suite = variant.get("env_suite", "kitchen")
    num_expl_envs = variant["num_expl_envs"]
    env_class_ = variant["env_class"]
    if env_suite == "metaworld":
        env_class = env_class_
        env_kwargs = {}
        env_fns = [
            lambda: TimeLimit(
                ImageEnvMetaworld(
                    make_env_multiworld(env_class_),
                    imwidth=64,
                    imheight=64,
                ),
                150,
            )
            for _ in range(num_expl_envs)
        ]
        expl_env = StableBaselinesVecEnv(env_fns=env_fns, start_method="fork")
        eval_env = ImageEnvMetaworld(
            make_env_multiworld(env_class_), imwidth=64, imheight=64
        )
        eval_env.reset()
        eval_env = DummyVecEnv([TimeLimit(eval_env, 150)], pass_render_kwargs=False)
        max_path_length = 151
    elif env_suite == "kitchen":
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
        elif env_class == "microwave_kettle_light_top_left_burner":
            env_class_ = KitchenMicrowaveKettleLightTopLeftBurnerV0
        elif env_class == "hinge_slide_bottom_left_burner_light":
            env_class_ = KitchenHingeSlideBottomLeftBurnerLightV0
        else:
            raise EnvironmentError("invalid env provided")

        env_fns = [
            lambda: TimeLimit(
                NormalizeActions(
                    ActionRepeat(
                        make_env(
                            env_class=env_class_,
                            env_kwargs=variant["env_kwargs"],
                        ),
                        2,
                    )
                ),
                500,
            )
            for _ in range(num_expl_envs)
        ]
        expl_env = StableBaselinesVecEnv(env_fns=env_fns, start_method="fork")

        eval_env = make_env(
            env_class=env_class_,
            env_kwargs=variant["env_kwargs"],
        )
        eval_env.reset()
        eval_env = ActionRepeat(eval_env, 2)
        eval_env = NormalizeActions(eval_env)
        eval_env = DummyVecEnv([TimeLimit(eval_env, 500)], pass_render_kwargs=False)
        max_path_length = 501

    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    world_model_class = WorldModel
    world_model = world_model_class(
        action_dim,
        image_shape=(3, 64, 64),
        **variant["model_kwargs"],
        env=eval_env,
    )
    actor_model_class = ActorModel
    actor = actor_model_class(
        variant["model_kwargs"]["model_hidden_size"],
        world_model.feature_size,
        hidden_activation=torch.nn.functional.elu,
        discrete_action_dim=0,
        continuous_action_dim=eval_env.action_space.low.size,
        env=eval_env,
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
        max_path_length,
        obs_dim,
        action_dim,
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
        eval_env,
        actor,
        vf,
        target_vf,
        world_model,
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
    trainer.pretrain_actor_vf(variant.get("num_actor_vf_pretrain_iters", 0))
    # algorithm.post_epoch_funcs.append(video_post_epoch_func)
    algorithm.to(ptu.device)
    algorithm.train()
    # video_post_epoch_func(algorithm, -1)
