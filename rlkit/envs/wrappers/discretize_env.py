import itertools

import numpy as np
from gym import Env
from gym.spaces import Discrete

from rlkit.envs.proxy_env import ProxyEnv


class DiscretizeEnv(ProxyEnv, Env):
    def __init__(self, wrapped_env, num_bins):
        super().__init__(wrapped_env)
        low = self.wrapped_env.action_space.low
        high = self.wrapped_env.action_space.high
        action_ranges = [
            np.linspace(low[i], high[i], num_bins)
            for i in range(len(low))
        ]
        self.idx_to_continuous_action = [
            np.array(x) for x in itertools.product(*action_ranges)
        ]
        self.action_space = Discrete(len(self.idx_to_continuous_action))

    def step(self, action):
        continuous_action = self.idx_to_continuous_action[action]
        return super().step(continuous_action)


