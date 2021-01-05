import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import Policy
from rlkit.torch.model_based.dreamer.basic_mcts_wm import UCT_search


class DiscreteMCTSPolicy(Policy):
    """"""

    def __init__(
        self,
        world_model,
        max_steps,
        num_primitives,
        action_dim,
        action_space,
        iterations,
        exploration_weight,
    ):
        self.world_model = world_model
        cont_action = action_space.low[num_primitives:]
        actions = np.array(
            [
                np.concatenate((np.squeeze(np.eye(num_primitives)[i]), cont_action))
                for i in range(num_primitives)
            ]
        )
        actions = ptu.from_numpy(actions)
        self.world_model.actions = actions
        self.max_steps = max_steps
        self.num_primitives = num_primitives
        self.action_dim = action_dim
        self.iterations = iterations
        self.exploration_weight = exploration_weight

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
        start_state, _ = self.world_model.obs_step(latent, action, embed)
        action = UCT_search(
            self.world_model,
            start_state,
            self.iterations,
            self.max_steps,
            self.num_primitives,
        )[0]
        action = self.world_model.actions[action].reshape(1, -1)
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
