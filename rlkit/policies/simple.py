import numpy as np

from rlkit.policies.base import SerializablePolicy


class ZeroPolicy(SerializablePolicy):
    """
    Policy that always outputs zero.
    """

    def __init__(self, action_dim):
        self.action_dim = action_dim

    def get_action(self, obs):
        return np.zeros(self.action_dim), {}


class RandomPolicy(SerializablePolicy):
    """
    Policy that always outputs zero.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        return self.action_space.sample(), {}
