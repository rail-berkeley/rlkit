from rlkit.policies.base import Policy
import numpy as np
import rlkit.torch.pytorch_util as ptu
import torch
from torch.distributions import Normal
import torch.nn.functional as F

class DreamerPolicy(Policy):
	"""
	"""

	def __init__(
			self,
			world_model,
			actor,
			obs_dim,
			action_dim,
			expl_amount=0.3,
			split_dist=False,
			split_size=0,
			exploration=False,
	):
		self.world_model = world_model
		self.actor = actor
		self.exploration = exploration
		self.obs_dim = obs_dim
		self.action_dim = action_dim
		self.expl_amount = expl_amount
		self.split_dist = split_dist
		self.split_size = split_size

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
		post, _, _, _, _, _, _ = self.world_model(observation.unsqueeze(0), action.unsqueeze(0))
		feat = self.world_model.get_feat(post).squeeze(0)
		dist = self.actor(feat)
		if self.exploration:
			action = dist.rsample()
			if self.split_dist:
				deter, cont, extra = action.split(self.split_size, -1)
				cont = torch.cat((cont, extra), -1)
				indices = torch.distributions.Categorical(logits=0*deter).sample()
				rand_action = F.one_hot(indices, deter.shape[-1])
				probs = ptu.rand(deter.shape[:1])
				deter = torch.where(probs.reshape(-1,1) < self.expl_amount, rand_action.int(), deter.int())
				cont = torch.clamp(Normal(cont, self.expl_amount).rsample(), -1, 1)
				action = torch.cat((deter, cont), -1)
			else:
				action = torch.clamp(Normal(action, self.expl_amount).rsample(), -1, 1)
		else:
			action = dist.mode()
		self.state = (latent, action)
		return ptu.get_numpy(action), {}

	def reset(self):
		self.state = None

class ActionSpaceSamplePolicy(Policy):
	def __init__(
			self,
			env
	):
		self.env = env

	def get_action(self, observation):
		return np.array([self.env.action_space.sample() for _ in range(self.env.n_envs)]), {}