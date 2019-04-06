from rlkit.policies.base import Policy


class RandomPolicy(Policy):
    """
    Policy that always outputs zero.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        return self.action_space.sample(), {}
