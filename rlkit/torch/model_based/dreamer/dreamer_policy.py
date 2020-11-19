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

    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """
        observation = ptu.from_numpy(np.array(observation))
        if self.state:
            latent, action = self.state
        else:
            latent = self.world_model.initial(observation.shape[0])
            action = ptu.zeros((observation.shape[0], self.action_dim))
        embed = self.world_model.encode(observation)
        latent, _ = self.world_model.obs_step(latent, action, embed)
        feat = self.world_model.get_feat(latent)
        dist = self.actor(feat)
        if self.exploration:
            action = dist.rsample()
            if self.discrete_continuous_dist:
                discrete, continuous = (
                    action[:, : self.discrete_action_dim],
                    action[:, self.discrete_action_dim :],
                )
                indices = torch.distributions.Categorical(logits=0 * discrete).sample()
                rand_action = F.one_hot(indices, discrete.shape[-1])
                probs = ptu.rand(discrete.shape[:1])
                # epsilon greedy
                discrete = torch.where(
                    probs.reshape(-1, 1) < self.expl_amount,
                    rand_action.int(),
                    discrete.int(),
                )
                continuous = torch.clamp(
                    Normal(continuous, self.expl_amount).rsample(), -1, 1
                )
                assert (discrete.sum(dim=1) == ptu.ones(discrete.shape[0])).all()
                action = torch.cat((discrete, continuous), -1)
            else:
                action = torch.clamp(
                    Normal(action.float(), self.expl_amount).rsample(), -1.4, 1.4
                )
        else:
            action = dist.mode()
        self.state = (latent, action)
        return ptu.get_numpy(action), {}

    def reset(self):
        self.state = None


class ActionSpaceSamplePolicy(Policy):
    def __init__(self, env):
        self.env = env

    def get_action(self, observation):
        return (
            np.array([self.env.action_space.sample() for _ in range(self.env.n_envs)]),
            {},
        )
