import numpy as np
from gym.spaces import Box, Dict

from rlkit.exploration_strategies.base import RawExplorationStrategy

class StubEnv(object):
    def __init__(self, obs_dim=1, action_dim=1, **kwargs):
        self.obs_dim = obs_dim
        obs_low = np.ones(obs_dim) * -1
        obs_high = np.ones(obs_dim)
        self._observation_space = Box(obs_low, obs_high)

        self.action_dim = action_dim
        action_low = np.ones(action_dim) * -1
        action_high = np.ones(action_dim)
        self._action_space = Box(action_low, action_high)

        print("stub env unused kwargs", kwargs)

    def reset(self):
        return np.zeros(self.obs_dim)

    def step(self, action):
        return np.zeros(self.obs_dim), 0, 0, {}

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return 99999

    @property
    def observation_space(self):
        return self._observation_space

class StubMultiEnv(object):
    def __init__(self, obs_dims=None, action_dim=1, **kwargs):
        self.obs_dims = obs_dims

        spaces = []
        for name in self.obs_dims:
            obs_dim = self.obs_dims[name]
            obs_low = np.ones(obs_dim) * -1
            obs_high = np.ones(obs_dim)
            spaces.append((name, Box(obs_low, obs_high)))
        self._observation_space = Dict(spaces)

        self.action_dim = action_dim
        action_low = np.ones(action_dim) * -1
        action_high = np.ones(action_dim)
        self._action_space = Box(action_low, action_high)

        print("stub env unused kwargs", kwargs)

    def reset(self):
        return self.get_obs()

    def step(self, action):
        return self.get_obs(), 0, 0, {}

    def get_obs(self):
        obs = dict()
        for name in self.obs_dims:
            obs_dim = self.obs_dims[name]
            obs[name] = np.zeros(obs_dim)
        return obs

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return 99999

    @property
    def observation_space(self):
        return self._observation_space


class StubPolicy(object):
    def __init__(self, action):
        self._action = action

    def get_action(self, *arg, **kwargs):
        return self._action, {}


class AddEs(RawExplorationStrategy):
    """
    return action + constant
    """
    def __init__(self, number):
        self._number = number

    def get_action(self, t, observation, policy, **kwargs):
        action, _ = policy.get_action(observation)
        return self.get_action_from_raw_action(action)

    def get_action_from_raw_action(self, action, **kwargs):
        return self._number + action
