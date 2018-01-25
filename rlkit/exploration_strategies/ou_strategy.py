import numpy as np
import numpy.random as nr

from rlkit.exploration_strategies.base import RawExplorationStrategy
from rlkit.core.serializable import Serializable


class OUStrategy(RawExplorationStrategy, Serializable):
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process

    Based on the rllab implementation.
    """

    def __init__(
            self,
            action_space,
            mu=0,
            theta=0.15,
            max_sigma=0.3,
            min_sigma=0.3,
            decay_period=100000,
    ):
        assert len(action_space.shape) == 1
        Serializable.quick_init(self, locals())
        if min_sigma is None:
            min_sigma = max_sigma
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self._max_sigma = max_sigma
        if min_sigma is None:
            min_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self.dim = np.prod(action_space.low.shape)
        self.low = action_space.low
        self.high = action_space.high
        self.state = np.ones(self.dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

    def get_action_from_raw_action(self, action, t=0, **kwargs):
        ou_state = self.evolve_state()
        self.sigma = (
            self._max_sigma
            - (self._max_sigma - self._min_sigma)
            * min(1.0, t * 1.0 / self._decay_period)
        )
        return np.clip(action + ou_state, self.low, self.high)

    def get_actions_from_raw_actions(self, actions, t=0, **kwargs):
        noise = (
            self.state + self.theta * (self.mu - self.state)
            + self.sigma * nr.randn(*actions.shape)
        )
        return np.clip(actions + noise, self.low, self.high)
