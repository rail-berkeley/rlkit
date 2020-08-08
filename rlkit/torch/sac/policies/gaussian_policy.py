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
from rlkit.torch.sac.policies.base import (
    TorchStochasticPolicy,
    PolicyFromDistributionGenerator,
    MakeDeterministic,
)

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# TODO: deprecate classes below in favor for PolicyFromDistributionModule


class TanhGaussianPolicyAdapter(TorchStochasticPolicy):
    """
    Usage:

    ```
    obs_processor = ...
    policy = TanhGaussianPolicyAdapter(obs_processor)
    ```
    """

    def __init__(
            self,
            obs_processor,
            obs_processor_output_dim,
            action_dim,
            hidden_sizes,
    ):
        super().__init__()
        self.obs_processor = obs_processor
        self.obs_processor_output_dim = obs_processor_output_dim
        self.mean_and_log_std_net = Mlp(
            hidden_sizes=hidden_sizes,
            output_size=action_dim * 2,
            input_size=obs_processor_output_dim,
        )
        self.action_dim = action_dim

    def forward(self, obs):
        h = self.obs_processor(obs)
        h = self.mean_and_log_std_net(h)
        mean, log_std = torch.split(h, self.action_dim, dim=1)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        tanh_normal = TanhNormal(mean, std)
        return tanh_normal


# noinspection PyMethodOverriding
class TanhGaussianPolicy(Mlp, TorchStochasticPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return TanhNormal(mean, std)

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob


class GaussianPolicy(Mlp, TorchStochasticPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture="shared",
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            output_activation=torch.tanh,
            **kwargs
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return MultivariateDiagonalNormal(mean, std)



class GaussianCNNPolicy(CNN, TorchStochasticPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture="shared",
            **kwargs
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
            init_w=init_w,
            output_activation=torch.tanh,
            **kwargs
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = super().forward(obs, return_last_activations=True)
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return MultivariateDiagonalNormal(mean, std)


class GaussianMixturePolicy(Mlp, TorchStochasticPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            num_gaussians=1,
            std_architecture="shared",
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim * num_gaussians,
            init_w=init_w,
            # output_activation=torch.tanh,
            **kwargs
        )
        self.action_dim = action_dim
        self.num_gaussians = num_gaussians
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]

            if self.std_architecture == "shared":
                self.last_fc_log_std = nn.Linear(last_hidden_size,
                                                 action_dim * num_gaussians)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim * num_gaussians, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX
        self.last_fc_weights = nn.Linear(last_hidden_size, num_gaussians)
        self.last_fc_weights.weight.data.uniform_(-init_w, init_w)
        self.last_fc_weights.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            # log_std = self.last_fc_log_std(h)
            # log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            # log_std = torch.sigmoid(self.last_fc_log_std(h))
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(self.std)
            log_std = self.log_std

        weights = F.softmax(self.last_fc_weights(h)).reshape(
            (-1, self.num_gaussians, 1))
        mixture_means = mean.reshape((-1, self.action_dim, self.num_gaussians,))
        mixture_stds = std.reshape((-1, self.action_dim, self.num_gaussians,))
        dist = GaussianMixture(mixture_means, mixture_stds, weights)
        return dist


class BinnedGMMPolicy(Mlp, TorchStochasticPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            num_gaussians=1,
            std_architecture="shared",
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim * num_gaussians,
            init_w=init_w,
            # output_activation=torch.tanh,
            **kwargs
        )
        self.action_dim = action_dim
        self.num_gaussians = num_gaussians
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]

            if self.std_architecture == "shared":
                self.last_fc_log_std = nn.Linear(last_hidden_size,
                                                 action_dim * num_gaussians)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim * num_gaussians, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX
        self.last_fc_weights = nn.Linear(last_hidden_size,
                                         action_dim * num_gaussians)
        self.last_fc_weights.weight.data.uniform_(-init_w, init_w)
        self.last_fc_weights.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        # preactivation = self.last_fc(h)
        # mean = self.output_activation(preactivation)
        if self.std is None:
            # log_std = self.last_fc_log_std(h)
            # log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            # log_std = torch.sigmoid(self.last_fc_log_std(h))
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(self.std)
            log_std = self.log_std
        batch_size = len(obs)
        logits = self.last_fc_weights(h).reshape(
            (-1, self.action_dim, self.num_gaussians,))
        weights = F.softmax(logits, dim=2)
        linspace = np.tile(np.linspace(-1, 1, self.num_gaussians),
                           (batch_size, self.action_dim, 1))
        mixture_means = ptu.from_numpy(linspace)
        mixture_stds = std.reshape((-1, self.action_dim, self.num_gaussians,))
        dist = GaussianMixtureFull(mixture_means, mixture_stds, weights)
        return dist


class TanhGaussianObsProcessorPolicy(TanhGaussianPolicy):
    def __init__(self, obs_processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_obs_dim = obs_processor.input_size
        self.pre_goal_dim = obs_processor.input_size
        self.obs_processor = obs_processor

    def forward(self, obs, *args, **kwargs):
        obs_and_goal = obs
        assert obs_and_goal.shape[1] == self.pre_obs_dim + self.pre_goal_dim
        obs = obs_and_goal[:, :self.pre_obs_dim]
        goal = obs_and_goal[:, self.pre_obs_dim:]

        h_obs = self.obs_processor(obs)
        h_goal = self.obs_processor(goal)

        flat_inputs = torch.cat((h_obs, h_goal), dim=1)
        return super().forward(flat_inputs, *args, **kwargs)


# noinspection PyMethodOverriding
class TanhCNNGaussianPolicy(CNN, TorchStochasticPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
            self,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            init_w=init_w,
            **kwargs
        )
        obs_dim = self.input_width * self.input_height
        action_dim = self.output_size
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(self.hidden_sizes) > 0:
                last_hidden_size = self.hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = super().forward(obs, return_last_activations=True)

        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std

        tanh_normal = TanhNormal(mean, std)
        return tanh_normal
