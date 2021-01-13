import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal, TransformedDistribution

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.model_based.dreamer.actor_models import SampleDist, TanhBijector
from rlkit.torch.model_based.dreamer.mlp import Mlp


class ConditionalContinuousActorModel(torch.nn.Module):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        env,
        discrete_action_dim=0,
        continuous_action_dim=0,
        hidden_activation=F.elu,
        min_std=1e-4,
        init_std=5.0,
        mean_scale=5.0,
        use_tanh_normal=True,
        use_per_primitive_actor=False,
        **kwargs
    ):
        super().__init__()
        self.env = env
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
                    hidden_sizes,
                    input_size=obs_dim,
                    output_size=len_v * 2,
                    hidden_activation=hidden_activation,
                    hidden_init=torch.nn.init.xavier_uniform_,
                    **kwargs
                )
                self.continuous_actor[k] = net
        else:
            self.continuous_actor = Mlp(
                hidden_sizes,
                input_size=obs_dim + discrete_action_dim,
                output_size=continuous_action_dim * 2,
                hidden_activation=hidden_activation,
                hidden_init=torch.nn.init.xavier_uniform_,
                **kwargs
            )
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim
        self._min_std = min_std
        self._init_std = ptu.tensor(init_std)
        self._mean_scale = mean_scale
        self.use_tanh_normal = use_tanh_normal
        self.raw_init_std = torch.log(torch.exp(self._init_std) - 1)

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
            h = torch.cat((observation, discrete_action), dim=1)
            cont_output = self.continuous_actor(h)
            mean, std = cont_output.split(self.continuous_action_dim, -1)
        if self.use_tanh_normal:
            mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
        std = F.softplus(std + self.raw_init_std) + self._min_std

        dist = Normal(mean, std)
        if self.use_tanh_normal:
            dist = TransformedDistribution(dist, TanhBijector())
        dist = Independent(dist, 1)
        dist = SampleDist(dist)
        return dist

    def compute_exploration_action(self, action, expl_amount):
        action = Normal(action, expl_amount).rsample()
        if self.use_tanh_normal:
            action = torch.clamp(action, -1, 1)
        return action
