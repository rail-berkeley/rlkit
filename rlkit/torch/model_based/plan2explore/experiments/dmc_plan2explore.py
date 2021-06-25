import gym
import numpy as np


class DeepMindControl:
    def __init__(self, name, size=(64, 64), camera=None):
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if isinstance(domain, str):
            from dm_control import suite

            self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

    @property
    def observation_space(self):

        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        obs["image"] = self.render()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs["image"] = self.render()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)


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
        o = o["image"].transpose(2, 0, 1).flatten()
        return o, r, d, i

    def reset(self):
        return self._env.reset()["image"].transpose(2, 0, 1).flatten()


def experiment(variant):
    import os

    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    import torch

    import rlkit.torch.pytorch_util as ptu
    from rlkit.envs.mujoco_vec_wrappers import DummyVecEnv
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
    from rlkit.torch.model_based.dreamer.world_models import WorldModel
    from rlkit.torch.model_based.plan2explore.latent_space_models import (
        OneStepEnsembleModel,
    )
    from rlkit.torch.model_based.plan2explore.plan2explore import Plan2ExploreTrainer
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
        env=eval_env,
    )
    actor_model_class = ActorModel
    eval_actor = actor_model_class(
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
        discrete_action_dim=0,
        continuous_action_dim=eval_env.action_space.low.size,
        env=eval_env,
        **variant["actor_kwargs"],
    )
    exploration_vf = Mlp(
        hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]] * 3,
        output_size=1,
        input_size=world_model.feature_size,
        hidden_activation=torch.nn.functional.elu,
    )
    exploration_target_vf = Mlp(
        hidden_sizes=[variant["model_kwargs"]["model_hidden_size"]] * 3,
        output_size=1,
        input_size=world_model.feature_size,
        hidden_activation=torch.nn.functional.elu,
    )
    variant["trainer_kwargs"]["exploration_target_vf"] = exploration_target_vf

    expl_policy = DreamerPolicy(
        world_model,
        exploration_actor,
        obs_dim,
        action_dim,
        exploration=True,
        expl_amount=variant.get("expl_amount", 0.3),
        discrete_action_dim=0,
        continuous_action_dim=eval_env.action_space.low.size,
    )
    eval_policy = DreamerPolicy(
        world_model,
        eval_actor,
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
        variant["replay_buffer_size"],
        expl_env,
        501,
        obs_dim,
        action_dim,
        replace=False,
        use_batch_length=True,
        batch_length=50,
    )
    trainer = Plan2ExploreTrainer(
        env=eval_env,
        world_model=world_model,
        actor=eval_actor,
        vf=vf,
        image_shape=(3, 64, 64),
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
    algorithm.to(ptu.device)
    algorithm.train()
