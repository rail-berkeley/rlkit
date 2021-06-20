import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import Policy


class DreamerPolicy(Policy):
    """"""

    def __init__(
        self,
        world_model,
        actor,
        obs_dim,
        action_dim,
        expl_amount=0.3,
        discrete_continuous_dist=False,
        discrete_action_dim=0,
        continuous_action_dim=0,
        exploration=False,
    ):
        self.world_model = world_model
        self.actor = actor
        self.exploration = exploration
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.expl_amount = expl_amount
        self.discrete_continuous_dist = discrete_continuous_dist
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim

    @torch.no_grad()
    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """
        with torch.cuda.amp.autocast():
            observation = ptu.from_numpy(np.array(observation))
            if self.state:
                prev_state, action = self.state
            else:
                prev_state = self.world_model.initial(observation.shape[0])
                action = ptu.zeros((observation.shape[0], self.action_dim))
            embed = self.world_model.encode(observation)
            new_state, _ = self.world_model.obs_step(prev_state, action, embed)
            feat = self.world_model.get_features(new_state)
            dist = self.actor(feat)
            action = dist.mode()
            if self.exploration:
                action = self.actor.compute_exploration_action(action, self.expl_amount)
            self.state = (new_state, action)
            return ptu.get_numpy(action), {}

    def reset(self, o):
        self.state = None


class ActionSpaceSamplePolicy(Policy):
    def __init__(self, env):
        self.env = env

    def get_action(self, observation):
        return (
            np.array([self.env.action_space.sample() for _ in range(self.env.n_envs)]),
            {},
        )

    def reset(self, o):
        return super().reset()
