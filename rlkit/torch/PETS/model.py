import torch
import random
from torch import nn as nn
import torch.nn.functional as F
import numpy as np

from rlkit.torch.networks import FlattenMlp
from rlkit.torch.core import torch_ify
from rlkit.torch.modules import swish


class Model(nn.Module):
    def __init__(self, hidden_sizes, obs_dim, action_dim, num_bootstrap, rew_function=None):
        '''
        Usage:
        model = Model(...)

        next_obs = model(obs, action)
        trajectory = model.unroll(obs, action_sequence)

        TODO: handle the different PETS sampling strategies.
        '''
        super().__init__()
        self.rew_function = rew_function
        self.predict_reward = self.rew_function is None
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.input_size = obs_dim + action_dim
        if self.predict_reward:
            self.output_dim = obs_dim * 2 + 1
        else:
            self.output_dim = obs_dim * 2

        self.num_bootstrap = num_bootstrap
        self._nets = nn.ModuleList()
        for i in range(num_bootstrap):
            # TODO: figure out what the network architecture should be
            self._nets.append(FlattenMlp(hidden_sizes, self.output_dim, self.input_size, hidden_activation=swish))
        self.max_logvar = nn.Parameter(torch.ones(1, self.obs_dim, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, self.obs_dim, dtype=torch.float32) * 10.0)
        self.trained_at_all = False

    def forward(self, obs, action, network_idx=None, return_net_outputs=False):
        # TODO: is this the usage I want?
        obs = torch_ify(obs)
        action = torch_ify(action)
        if network_idx is None:
            network = random.choice(self._nets)
        else:
            network = self._nets[network_idx]
        output = network(obs, action)
        # TODO: possibly wrap this in a Probabilistic Network class
        mean = output[:, :self.obs_dim]
        logvar = output[:, self.obs_dim:2*self.obs_dim]
        if self.predict_reward:
            reward = output[:, -1:]
        # do variance pinning
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        var = torch.exp(logvar)
        # sampling trick
        next_obs = mean + torch.randn_like(mean) * var.sqrt()

        if not self.predict_reward:
            reward = self.rew_function(obs, action, next_obs)
        if return_net_outputs:
            return mean, logvar, reward
        return next_obs, reward

    def unroll(self, obs, action_sequence, sampling_strategy):
        '''
        obs: batch_size * obs_dim (Tensor)
        action_sequence: batch_size * timesteps * action_dim (Tensor)
        sampling_strategy: one of "TS1" or "TSinf"
        return
        observations: batch_size * timesteps * obs_dim
        rewards: batch_size * timesteps
        '''
        obs = torch_ify(obs)
        action_sequence = torch_ify(action_sequence)
        batch_size = action_sequence.shape[0]
        n_timesteps = action_sequence.shape[1]
        obs_output = []
        rew_output = []
        # sampling the initial bootstrap assignments for every particle
        bootstrap_assignments = np.random.randint(self.num_bootstrap, size=batch_size)
        for i in range(n_timesteps):
            bs_obs = []
            bs_rew = []
            # we just run all networks on all inputs and then do the sampling after
            for network_idx in range(self.num_bootstrap):
                # rely on the fact that self.forward() is probabilistic and stuff
                next_obs, reward = self.forward(obs, action_sequence[:, i, :], network_idx)
                bs_obs.append(next_obs)
                bs_rew.append(reward)
            bs_obs = torch.stack(bs_obs)
            bs_rew = torch.stack(bs_rew)
            # do the sampling for the bootstrap
            next_obs = bs_obs[bootstrap_assignments, range(batch_size), :]
            next_rew = bs_rew[bootstrap_assignments, range(batch_size)]

            # move observation forward
            obs = next_obs
            obs_output.append(next_obs)
            rew_output.append(next_rew)
            # resample if needed
            if sampling_strategy == 'TS1':
                bootstrap_assignments = np.random.randint(self.num_bootstrap, size=batch_size)
        observations = torch.stack(obs_output, dim=1)
        rewards = torch.stack(rew_output, dim=1)
        return observations, rewards

    def bound_loss(self):
        return 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())


def gaussian_log_loss(mean, logvar, targets):
    inv_var = torch.exp(-logvar)
    loss = ((mean - targets) ** 2) * inv_var + logvar
    return loss.sum()
