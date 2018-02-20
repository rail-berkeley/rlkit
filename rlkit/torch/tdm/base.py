import abc

from rlkit.policies.base import SerializablePolicy


class UniversalPolicy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action(self, observation, goal, tau, **kwargs):
        pass

    def reset(self):
        pass

    def get_param_values(self):
        return None

    def set_param_values(self, param_values):
        return


class RandomUniversalPolicy(UniversalPolicy, SerializablePolicy):
    """
    Policy that always outputs zero.
    """

    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def get_action(self, *args, **kwargs):
        return self.action_space.sample(), {}
