import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal, TransformedDistribution

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.model_based.dreamer.actor_models import (
    SafeTruncatedNormal,
    SampleDist,
    TanhBijector,
)
from rlkit.torch.model_based.dreamer.mlp import Mlp


class ConditionalContinuousActorModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        obs_dim,
        env,
        num_layers=4,
        discrete_action_dim=0,
        continuous_action_dim=0,
        hidden_activation=F.elu,
        min_std=1e-4,
        init_std=5.0,
        mean_scale=5.0,
        use_tanh_normal=True,
        use_per_primitive_actor=False,
        discrete_continuous_dist=False,
        dist="tanh_normal_dreamer_v1",
        **kwargs,
    ):
        super().__init__()
        self.env = env
        self.discrete_continuous_dist = discrete_continuous_dist
        self.use_per_primitive_actor = use_per_primitive_actor
        if self.use_per_primitive_actor:
            self.continuous_actor = torch.nn.ModuleDict()
            for (
                k,
                v,
            ) in env.primitive_name_to_action_idx.items():
                if type(v) == int:
                    len_v = 1
                else:
                    len_v = len(v)
                net = Mlp(
                    [hidden_size] * num_layers,
                    input_size=obs_dim,
                    output_size=len_v * 2,
                    hidden_activation=hidden_activation,
                    hidden_init=torch.nn.init.xavier_uniform_,
                    **kwargs,
                )
                self.continuous_actor[k] = net
        else:
            self.continuous_actor = Mlp(
                [hidden_size] * num_layers,
                input_size=obs_dim + discrete_action_dim,
                output_size=continuous_action_dim * 2,
                hidden_activation=hidden_activation,
                hidden_init=torch.nn.init.xavier_uniform_,
                **kwargs,
            )
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim
        self._min_std = min_std
        self._init_std = ptu.tensor(init_std)
        self._mean_scale = mean_scale
        self.use_tanh_normal = use_tanh_normal
        self.raw_init_std = torch.log(torch.exp(self._init_std) - 1)
        self._dist = dist

    def forward(self, inputs):
        discrete_action, observation = inputs
        if self.use_per_primitive_actor:
            primitive_indices = torch.argmax(discrete_action, dim=1)
            mean = ptu.zeros((primitive_indices.shape[0], self.continuous_action_dim))
            std = ptu.zeros((primitive_indices.shape[0], self.continuous_action_dim))
            for k, v in self.env.primitive_name_to_action_idx.items():
                if type(v) is int:
                    v = [v]
                if len(v) < 1:
                    continue
                output = self.continuous_actor[k](observation)
                mean = mean.type(output.dtype)
                std = std.type(output.dtype)
                mean_, std_ = output.split(len(v), -1)
                mean[:, v] = mean_
                std[:, v] = std_
        else:
            h = torch.cat((observation, discrete_action), dim=-1)
            cont_output = self.continuous_actor(h)
            mean, std = cont_output.split(self.continuous_action_dim, -1)
        raw_init_std = self.raw_init_std
        action_mean, action_std = mean, std
        if self._dist == "tanh_normal_dreamer_v1":
            action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
            action_std = F.softplus(action_std + raw_init_std) + self._min_std

            dist = Normal(action_mean, action_std)
            dist = TransformedDistribution(dist, TanhBijector())
            dist = Independent(dist, 1)
            dist = SampleDist(dist)
        elif self._dist == "tanh_normal":
            action_mean = torch.tanh(action_mean)
            action_std = F.softplus(action_std + self._init_std) + self._min_std
            dist = Normal(action_mean, action_std)
            dist = TransformedDistribution(dist, TanhBijector())
            dist = Independent(dist, 1)
            dist = SampleDist(dist)
        elif self._dist == "tanh_normal_5":
            action_mean = 5 * torch.tanh(action_mean / 5)
            action_std = F.softplus(action_std + 5) + 5

            dist = Normal(action_mean, action_std)
            dist = TransformedDistribution(dist, TanhBijector())
            dist = Independent(dist, 1)
            dist = SampleDist(dist)
        elif self._dist == "trunc_normal":
            action_mean = torch.tanh(action_mean)
            action_std = 2 * torch.sigmoid(action_std / 2) + self._min_std
            dist = SafeTruncatedNormal(action_mean, action_std, -1, 1)
            dist = Independent(dist, 1)
        return dist

    def compute_exploration_action(self, action, expl_amount):
        action = Normal(action, expl_amount).rsample()
        if self.use_tanh_normal:
            action = torch.clamp(action, -1, 1)
        return action
