import os
import pickle

import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import Policy


class DreamerPolicy(Policy):
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
    @torch.cuda.amp.autocast()
    def get_action(
        self,
        observation,
        use_raps_obs=False,
        use_true_actions=True,
        use_obs=True,
    ):
        """
        :param observation:
        :return: action, debug_dictionary
        """
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
        return ptu.get_numpy(action), {"state": new_state}

    def reset(self, o):
        self.state = None

    def save(self, path, suffix):
        world_model = self.world_model
        actor = self.actor

        delattr(self, "world_model")
        delattr(self, "actor")

        pickle.dump(self, open(os.path.join(path, suffix), "wb"))

        base_suffix = suffix.replace(".pkl", "")
        torch.save(
            {
                "actor_state_dict": actor.state_dict(),
                "world_model_state_dict": world_model.state_dict(),
            },
            os.path.join(path, base_suffix + "_networks.ptc"),
        )

        self.world_model = world_model
        self.actor = actor

    def load(self, path, suffix):
        policy = pickle.load(open(os.path.join(path, suffix), "rb"))
        policy.world_model = self.world_model
        policy.actor = self.actor
        base_suffix = suffix.replace(".pkl", "")
        checkpoint = torch.load(os.path.join(path, base_suffix + "_networks.ptc"))
        policy.actor.load_state_dict(checkpoint["actor_state_dict"])
        policy.world_model.load_state_dict(checkpoint["world_model_state_dict"])
        return policy


class DreamerLowLevelRAPSPolicy(DreamerPolicy):
    def __init__(
        self, *args, num_low_level_actions_per_primitive, low_level_action_dim, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_low_level_actions_per_primitive = num_low_level_actions_per_primitive
        self.low_level_action_dim = low_level_action_dim

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def get_action(
        self,
        observation,
        use_raps_obs=False,
        use_true_actions=True,
        use_obs=True,
    ):
        """
        :param observation:
        :return: action, debug_dictionary
        """
        low_level_action, low_level_obs = observation
        observation = ptu.from_numpy(low_level_obs)
        if self.state:
            low_level_action = ptu.from_numpy(low_level_action)
            assert (
                observation.shape[1] == self.num_low_level_actions_per_primitive
            ), f"{observation.shape}, {self.num_low_level_actions_per_primitive}"
            assert (
                low_level_action.shape[1] == self.num_low_level_actions_per_primitive
            ), f"{low_level_action.shape}, {self.num_low_level_actions_per_primitive}"
            new_state, low_level_action_pred = self.world_model.forward_high_level_step(
                self.state[0],
                observation,
                low_level_action,
                self.num_low_level_actions_per_primitive,
                self.state[1],
                use_raps_obs,
                use_true_actions,
                use_obs,
            )
            low_level_action_pred = torch.cat(low_level_action_pred, axis=0)
        else:
            prev_state = self.world_model.initial(observation.shape[0])
            embed = self.world_model.encode(observation)
            new_state, _ = self.world_model.obs_step(
                prev_state,
                ptu.zeros((observation.shape[0], self.low_level_action_dim)),
                embed,
            )
            low_level_action_pred = ptu.zeros(
                (observation.shape[0], self.low_level_action_dim)
            )
        feat = self.world_model.get_features(new_state)
        dist = self.actor(feat)
        action = dist.mode()
        if self.exploration:
            action = self.actor.compute_exploration_action(action, self.expl_amount)
        self.state = (new_state, action)
        return ptu.get_numpy(action), {
            "state": new_state,
            "low_level_action_pred": ptu.get_numpy(low_level_action_pred),
        }


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

    def save(self, path, suffix):
        env = self.env
        delattr(self, "env")
        pickle.dump(self, open(os.path.join(path, suffix), "wb"))
        base_suffix = suffix.replace(".pkl", "")
        env.save(path, base_suffix + "_env.pkl")
        self.env = env

    def load(self, path, suffix):
        policy = pickle.load(open(os.path.join(path, suffix), "rb"))
        base_suffix = suffix.replace(".pkl", "")
        policy.env = self.env.load(path, base_suffix + "_env.pkl")
