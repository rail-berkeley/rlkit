from rlkit.policies.base import Policy
import numpy as np
import rlkit.torch.pytorch_util as ptu
import torch
import torch.functional as F

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
			exploration=False,
	):
		self.world_model = world_model
		self.actor = actor
		self.exploration = exploration
		self.obs_dim = obs_dim
		self.action_dim = action_dim
		self.expl_amount = expl_amount

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
			action = torch.zeros((observation.shape[0], self.action_dim))
		embed = self.world_model.encode(observation)
		latent, _ = self.world_model.obs_step(latent, action, embed)
		feat = self.world_model.get_feat(latent)
		if self.exploration:
			action = self.actor(feat).sample()
			action = torch.clamp(torch.normal(action, self.expl_amount), -1, 1)
		else:
			action = self.actor(feat).mode()
		self.state = (latent, action)
		return ptu.get_numpy(action), {}

	def reset(self):
		self.state = None