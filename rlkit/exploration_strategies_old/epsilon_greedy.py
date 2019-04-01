import random

from gym.spaces import Discrete

from rlkit.exploration_strategies.base import RawExplorationStrategy
from rlkit.core.serializable import Serializable


class EpsilonGreedy(RawExplorationStrategy, Serializable):
    """
    Take a random discrete action with some probability.
    """
    def __init__(self, action_space, prob_random_action=0.1):
        Serializable.quick_init(self, locals())
        assert isinstance(action_space, Discrete)
        Serializable.quick_init(self, locals())
        self.prob_random_action = prob_random_action
        self.action_space = action_space

    def get_action_from_raw_action(self, action, **kwargs):
        if random.random() <= self.prob_random_action:
            return self.action_space.sample()
        return action
