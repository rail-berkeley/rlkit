import random
import numpy as np
from rlkit.exploration_strategies.base import RawExplorationStrategy
from rlkit.util.ml_util import LinearSchedule


class EpsilonGreedy(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """
    def __init__(self, action_space, action_prior=None, prob_init=0.1, prob_end=0.1, steps=1e6):
        self.prob_random_action = LinearSchedule(prob_init, prob_end, steps)
        self.action_space = action_space
        self.action_prior = action_prior

        assert (self.action_prior is None) or len(self.action_prior) == self.action_space.n

    def get_exploration_action(self, action, **kwargs):
        if self.action_prior is not None:
            a = np.random.dirichlet(self.action_prior, size=1).argmax()
        else:
            a = self.action_space.sample()
        return a

    def get_action_from_raw_action(self, action,t=0):
        if random.random() <= self.prob_random_action.get_value(t):
            return self.get_exploration_action(action)
        return action
