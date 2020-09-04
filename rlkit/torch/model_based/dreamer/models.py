from rlkit.torch.core import PyTorchModule
from rlkit.torch.model_based.dreamer.conv_networks import CNN, DCNN
from rlkit.torch.networks import Mlp
from torch.distributions import Normal
import torch.nn.functional as F
from torch.distributions.transformed_distribution import TransformedDistribution
import torch
import numpy as np
import rlkit.torch.pytorch_util as ptu

class ActorModel(Mlp):
	def __init__(self,
				 hidden_sizes,
				 obs_dim,
				 action_dim,
				 init_w=1e-3,
				 hidden_activation=F.elu,
				 min_std=1e-4, init_std=5, mean_scale=5,
				 **kwargs):
		super().__init__(hidden_sizes,
						 input_size=obs_dim,
						 output_size=action_dim*2,
						 init_w=init_w,
						 hidden_activation=hidden_activation,
						 **kwargs)
		self.action_dim = action_dim
		self._min_std = min_std
		self._init_std = ptu.from_numpy(np.array(init_std))
		self._mean_scale = mean_scale

	def forward(self, feat):
		raw_init_std = torch.log(torch.exp(self._init_std) - 1)
		h = feat
		for i, fc in enumerate(self.fcs):
			h = self.hidden_activation(fc(h))
		last = self.last_fc(h)
		action_mean, action_std_dev = last.split(self.action_dim, -1)
		action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
		action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std
		dist = Normal(action_mean, action_std)
		dist = TransformedDistribution(dist, TanhBijector())
		dist = torch.distributions.Independent(dist,1)
		dist = SampleDist(dist)
		return dist

# "atanh", "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def atanh(x):
	return 0.5 * torch.log((1 + x) / (1 - x))

class TanhBijector(torch.distributions.Transform):
	def __init__(self):
		super().__init__()
		self.bijective = True

	@property
	def sign(self): return 1.

	def _call(self, x): return torch.tanh(x)

	def _inverse(self, y: torch.Tensor):
		y = torch.where(
			(torch.abs(y) <= 1.),
			torch.clamp(y, -0.99999997, 0.99999997),
			y)
		y = atanh(y)
		return y

	def log_abs_det_jacobian(self, x, y):
		return 2. * (np.log(2) - x - F.softplus(-2. * x))


class SampleDist:
	def __init__(self, dist, samples=100):
		self._dist = dist
		self._samples = samples

	@property
	def name(self):
		return 'SampleDist'

	def __getattr__(self, name):
		return getattr(self._dist, name)

	def mean(self):
		sample = self._dist.rsample()
		return torch.mean(sample, 0)

	def mode(self):
		dist = self._dist.expand((self._samples, *self._dist.batch_shape))
		sample = dist.rsample()
		logprob = dist.log_prob(sample)
		batch_size = sample.size(1)
		feature_size = sample.size(2)
		indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
		return torch.gather(sample, 0, indices).squeeze(0)

	def entropy(self):
		dist = self._dist.expand((self._samples, *self._dist.batch_shape))
		sample = dist.rsample()
		logprob = dist.log_prob(sample)
		return -torch.mean(logprob, 0)

	def sample(self):
		return self._dist.sample()

class WorldModel(PyTorchModule):
	def __init__(
			self,
			action_dim,
			stochastic_state_size=60,
			deterministic_state_size=400,
			embedding_size=1024,
			model_hidden_size=400, model_act=F.elu,
			depth=32, conv_act=F.relu,
			reward_num_layers=2,
				 ):
		super().__init__()
		self.obs_step_mlp = Mlp(
			hidden_sizes=[model_hidden_size],
			input_size=embedding_size+deterministic_state_size,
			output_size=2*stochastic_state_size,
			hidden_activation=model_act,
		)
		self.img_step_layer = torch.nn.Linear(stochastic_state_size+action_dim, deterministic_state_size)
		self.img_step_mlp = Mlp(
			hidden_sizes=[model_hidden_size],
			input_size=deterministic_state_size,
			output_size=2*stochastic_state_size,
			hidden_activation=model_act,
		)
		self.rnn = torch.nn.GRUCell(deterministic_state_size, deterministic_state_size)
		self.conv_encoder = CNN(
			64, 64, 3, embedding_size, [4]*4, [depth, depth*2, depth*4, depth*8], [2]*4, [0]*4, hidden_activation=conv_act, use_last_fc=False,
		)

		self.conv_decoder = DCNN(
			stochastic_state_size+deterministic_state_size, [],
			1, 1, depth*32, 6, 2, 3, [5, 5, 6], [depth*4, depth*2, depth*1], [2]*3, [0]*3, hidden_activation=conv_act,
							 )

		self.reward = Mlp(
			hidden_sizes=[model_hidden_size]*reward_num_layers,
			input_size=stochastic_state_size+deterministic_state_size,
			output_size=1,
			hidden_activation=model_act,
		)

		self.model_act=model_act
		self.feature_size = stochastic_state_size + deterministic_state_size
		self.stochastic_state_size = stochastic_state_size
		self.deterministic_state_size = deterministic_state_size
		self.modules = [self.obs_step_mlp, self.img_step_layer, self.img_step_mlp, self.rnn,self.conv_encoder, self.reward, self.conv_decoder]

	def obs_step(self, prev_state, prev_action, embed):
		prior = self.img_step(prev_state, prev_action)
		x = torch.cat([prior['deter'], embed], -1)
		x = self.obs_step_mlp(x)
		mean, std = x.split(self.stochastic_state_size, -1)
		std = F.softplus(std) + 0.1
		stoch = self.get_dist(mean, std).sample()
		post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
		return post, prior

	def img_step(self, prev_state, prev_action):
		x = torch.cat([prev_state['stoch'], prev_action], -1)
		x = self.model_act(self.img_step_layer(x))
		deter = self.rnn(x, prev_state['deter'])
		x = self.img_step_mlp(deter)
		mean, std = x.split(self.stochastic_state_size, -1)
		std = F.softplus(std) + 0.1
		stoch = self.get_dist(mean, std).sample()
		prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
		return prior

	def forward_batch(self, obs, action, state=None,):
		embed = self.encode(obs)
		posterior_params, prior_params = self.obs_step(state, action, embed)
		feat = self.get_feat(posterior_params)
		image_params = self.decode(feat)
		reward_params = self.reward(feat)
		return posterior_params, prior_params, image_params, reward_params

	def forward(self, obs, action, state=None, loop_through_path_length=False):
		original_batch_size = obs.shape[0]
		if not state:
			state = self.initial(original_batch_size)
		if loop_through_path_length:
			path_length = obs.shape[1]
			post_full, prior_full = dict(mean=[], std=[], stoch=[], deter=[]), dict(mean=[], std=[], stoch=[], deter=[])
			image_full, reward_full = [], []

			for i in range(path_length):
				posterior_params, prior_params, image_params, reward_params = self.forward_batch(obs[:, i], action[:, i], state)
				image_full.append(image_params)
				reward_full.append(reward_params)
				for k in post_full.keys():
					post_full[k].append(posterior_params[k].reshape((1, posterior_params[k].shape[0], posterior_params[k].shape[1] )))

				for k in prior_full.keys():
					prior_full[k].append(prior_params[k].reshape((1, prior_params[k].shape[0], prior_params[k].shape[1] )))
				state = posterior_params

			image_full = torch.cat(image_full)
			reward_full = torch.cat(reward_full)
			for k in post_full.keys():
				post_full[k] = torch.cat(post_full[k]).permute(1, 0, 2)

			for k in prior_full.keys():
				prior_full[k] = torch.cat(prior_full[k]).permute(1, 0, 2)
			return post_full, prior_full, self.get_dist(post_full['mean'], post_full['std']), self.get_dist(prior_full['mean'], prior_full['std']), self.get_dist(image_full, ptu.ones_like(image_full), dims=3), self.get_dist(reward_full, ptu.ones_like(reward_full))
		else:
			posterior_params, prior_params, image_params, reward_params = self.forward_batch(obs, action, state)
			return self.get_dist(posterior_params['mean'], posterior_params['std']), self.get_dist(prior_params['mean'], prior_params['std']), self.get_dist(image_params, ptu.ones_like(image_params), dims=3), self.get_dist(reward_params, ptu.ones_like(reward_params))

	def get_feat(self, state):
		return torch.cat([state['stoch'], state['deter']], -1)

	def get_dist(self, mean, std, dims=1):
		return torch.distributions.Independent(torch.distributions.Normal(mean, std), dims)

	def encode(self, obs):
		return self.conv_encoder(self.preprocess(obs))

	def decode(self, feat):
		return self.conv_decoder(feat)

	def preprocess(self, obs):
		obs= obs / 255.0 - 0.5
		return obs

	def initial(self, batch_size):
		return dict(
			mean=ptu.zeros([batch_size, self.stochastic_state_size]),
			std=ptu.zeros([batch_size, self.stochastic_state_size]),
			stoch=ptu.zeros([batch_size, self.stochastic_state_size]),
			deter=ptu.zeros([batch_size, self.deterministic_state_size])
		)
