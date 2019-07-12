import random
import numpy as np
from rlkit.exploration_strategies.base import RawExplorationStrategy
from rlkit.util.ml_util import LinearSchedule


class EpsilonGreedy(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """
    def __init__(self, action_space, prob_init=0.1, prob_end=0.1, steps=1e6):
        self.prob_random_action = LinearSchedule(prob_init, prob_end, steps)
        self.action_space = action_space

    def get_action_from_raw_action(self, action,t=0):
        if random.random() <= self.prob_random_action.get_value(t):
            return self.action_space.sample()
        return action
