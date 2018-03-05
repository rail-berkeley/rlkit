import abc

from rlkit.policies.base import ExplorationPolicy, SerializablePolicy


class ExplorationStrategy(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action(self, t, observation, policy, **kwargs):
        pass

    @abc.abstractmethod
    def get_actions(self, t, observation, policy, **kwargs):
        pass

    def reset(self):
        pass


class RawExplorationStrategy(ExplorationStrategy, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action_from_raw_action(self, action, **kwargs):
        pass

    def get_actions_from_raw_actions(self, actions, **kwargs):
        raise NotImplementedError()

    def get_action(self, t, policy, *args, **kwargs):
        action, agent_info = policy.get_action(*args, **kwargs)
        return self.get_action_from_raw_action(action, t=t), agent_info

    def get_actions(self, t, observation, policy, **kwargs):
        actions = policy.get_actions(observation)
        return self.get_actions_from_raw_actions(actions, t=t, **kwargs)

    def reset(self):
        pass


class PolicyWrappedWithExplorationStrategy(ExplorationPolicy, SerializablePolicy):
    def __init__(
            self,
            exploration_strategy: ExplorationStrategy,
            policy: SerializablePolicy,
    ):
        self.es = exploration_strategy
        self.policy = policy
        self.t = 0

    def set_num_steps_total(self, t):
        self.t = t

    def get_action(self, *args, **kwargs):
        return self.es.get_action(self.t, self.policy, *args, **kwargs)

    def get_actions(self, *args, **kwargs):
        return self.es.get_actions(self.t, self.policy, *args, **kwargs)

    def reset(self):
        self.es.reset()
        self.policy.reset()

    def get_param_values(self):
        return self.policy.get_param_values()

    def set_param_values(self, param_values):
        self.policy.set_param_values(param_values)

    def get_param_values_np(self):
        return self.policy.get_param_values_np()

    def set_param_values_np(self, param_values):
        self.policy.set_param_values_np(param_values)
