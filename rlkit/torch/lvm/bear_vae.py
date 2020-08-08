import abc
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy
from rlkit.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull,
)
from rlkit.torch.networks import Mlp, CNN
from rlkit.torch.networks.basic import MultiInputSequential
from rlkit.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)
from rlkit.torch.lvm.latent_variable_model import LatentVariableModel


class VAEPolicy(LatentVariableModel):
    def __init__(
            self,
            obs_dim,
            action_dim,
            latent_dim,
    ):
        encoder = Encoder(obs_dim, latent_dim, action_dim)
        decoder = Decoder(obs_dim, latent_dim, action_dim)
        super().__init__(encoder, decoder)

        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.encoder.e1(torch.cat([state, action], 1)))
        z = F.relu(self.encoder.e2(z))

        mean = self.encoder.mean(z)
        # Clamped for numerical stability
        log_std = self.encoder.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * ptu.from_numpy(
            np.random.normal(0, 1, size=(std.size())))

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = ptu.from_numpy(np.random.normal(0, 1, size=(
            state.size(0), self.latent_dim))).clamp(-0.5, 0.5)

        a = F.relu(self.decoder.d1(torch.cat([state, z], 1)))
        a = F.relu(self.decoder.d2(a))
        return torch.tanh(self.decoder.d3(a))

    def decode_multiple(self, state, z=None, num_decode=10):
        if z is None:
            z = ptu.from_numpy(np.random.normal(0, 1, size=(
            state.size(0), num_decode, self.latent_dim))).clamp(-0.5, 0.5)

        a = F.relu(self.decoder.d1(torch.cat(
            [state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z],
            2)))
        a = F.relu(self.decoder.d2(a))
        return torch.tanh(self.decoder.d3(a)), self.decoder.d3(a)


class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, action_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.e1 = torch.nn.Linear(obs_dim + action_dim, 750)
        self.e2 = torch.nn.Linear(750, 750)

        self.mean = torch.nn.Linear(750, self.latent_dim)
        self.log_std = torch.nn.Linear(750, self.latent_dim)

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        return MultivariateDiagonalNormal(mean, std)


class Decoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, action_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.d1 = torch.nn.Linear(obs_dim + self.latent_dim, 750)
        self.d2 = torch.nn.Linear(750, 750)
        self.d3 = torch.nn.Linear(750, action_dim)

    def forward(self, state, z):
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return Delta(torch.tanh(self.d3(a)))
