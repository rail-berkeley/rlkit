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


class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class ActionRepeat:
    def __init__(self, env, amount):
        self._env = env
        self._amount = amount

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
            obs, reward, done, info = self._env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class NormalizeActions:
    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    @property
    def observation_space(self):
        return gym.spaces.Box(0, 255, (64 * 64 * 3,), dtype=np.uint8)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        o, r, d, i = self._env.step(original)
        return o, r, d, i

    def reset(self):
        return self._env.reset()


def run_experiment(variant):

    import os

    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

    import torch
    from d4rl.kitchen.kitchen_envs import (
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
    from rlkit.torch.model_based.dreamer.mcts_policy import DiscreteMCTSPolicy
    from rlkit.torch.model_based.dreamer.mlp import Mlp
    from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
    from rlkit.torch.model_based.dreamer.world_models import (
        StateConcatObsWorldModel,
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
    else:
        raise EnvironmentError("invalid env provided")

    expl_env = make_env(
        env_class=env_class_,
        env_kwargs=variant["env_kwargs"],
    )
    expl_env.reset()
    expl_env = ActionRepeat(expl_env, 2)
    expl_env = NormalizeActions(expl_env)
    expl_env = DummyVecEnv([TimeLimit(expl_env, 500)], pass_render_kwargs=False)

    eval_env = make_env(
        env_class=env_class_,
        env_kwargs=variant["env_kwargs"],
    )
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
        501,
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