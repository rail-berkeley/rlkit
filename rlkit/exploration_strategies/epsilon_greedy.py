import random

from rlkit.exploration_strategies.base import RawExplorationStrategy
from rlkit.util.ml_util import LinearSchedule


class EpsilonGreedy(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """
    def __init__(self, action_space, prob_random_action=0.1, prob_end=None, steps=1e6):
        """
        If prob_end is None, this will default to a fixed schedule. 
        """
        if prob_end is not None:
            self.prob_random_action = LinearSchedule(prob_random_action, prob_end, steps)
        else:
            self.prob_random_action = LinearSchedule(prob_random_action, prob_random_action, steps)
        self.action_space = action_space

    def get_action_from_raw_action(self, action,t=0 **kwargs):
        if random.random() <= self.prob_random_action.get_value(t):
            return self.action_space.sample()
        return action
