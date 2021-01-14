import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal, Transform, TransformedDistribution
from torch.distributions.one_hot_categorical import OneHotCategorical

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.model_based.dreamer.mlp import Mlp
from rlkit.torch.model_based.dreamer.truncated_normal import TruncatedNormal


class ActorModel(Mlp):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        env,
        use_per_primitive_actor=False,
        discrete_continuous_dist=False,
        discrete_action_dim=0,
        continuous_action_dim=0,
        hidden_activation=F.elu,
        min_std=1e-4,
        init_std=5.0,
        mean_scale=5.0,
        use_tanh_normal=True,
        **kwargs
    ):
        self.discrete_continuous_dist = discrete_continuous_dist
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim
        if self.discrete_continuous_dist:
            self.output_size = self.discrete_action_dim + self.continuous_action_dim * 2
        else:
            self.output_size = self.continuous_action_dim * 2
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=self.output_size,
            hidden_activation=hidden_activation,
            hidden_init=torch.nn.init.xavier_uniform_,
            **kwargs
        )
        self._min_std = min_std
        self._init_std = ptu.tensor(init_std)
        self._mean_scale = mean_scale
        self.use_tanh_normal = use_tanh_normal

    def forward(self, input):
        raw_init_std = torch.log(torch.exp(self._init_std) - 1)
        h = input
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        last = self.last_fc(h)
        if self.discrete_continuous_dist:
            assert last.shape[1] == self.output_size
            mean, continuous_action_std = (
                last[:, : self.discrete_action_dim + self.continuous_action_dim],
                last[:, self.discrete_action_dim + self.continuous_action_dim :],
            )
            discrete_logits, continuous_action_mean, extra = mean.split(
                self.discrete_action_dim, -1
            )
            continuous_action_mean = torch.cat((continuous_action_mean, extra), -1)

            dist1 = OneHotDist(logits=discrete_logits)

            if self.use_tanh_normal:
                # why is this done???
                continuous_action_mean = self._mean_scale * torch.tanh(
                    continuous_action_mean / self._mean_scale
                )
            continuous_action_std = (
                F.softplus(continuous_action_std + raw_init_std) + self._min_std
            )

            dist2 = Normal(continuous_action_mean, continuous_action_std)
            if self.use_tanh_normal:
                dist2 = TransformedDistribution(dist2, TanhBijector())
            dist2 = Independent(dist2, 1)
            dist2 = SampleDist(dist2)
            dist = SplitDist(dist1, dist2)
        else:
            action_mean, action_std = last.split(self.continuous_action_dim, -1)
            if self.use_tanh_normal:
                action_mean = self._mean_scale * torch.tanh(
                    action_mean / self._mean_scale
                )
            action_std = F.softplus(action_std + raw_init_std) + self._min_std

            dist = Normal(action_mean, action_std)
            if self.use_tanh_normal:
                dist = TransformedDistribution(dist, TanhBijector())
            dist = torch.distributions.Independent(dist, 1)
            dist = SampleDist(dist)
        return dist

    def compute_exploration_action(self, action, expl_amount):
        if self.discrete_continuous_dist:
            discrete, continuous = (
                action[:, : self.discrete_action_dim],
                action[:, self.discrete_action_dim :],
            )
            indices = torch.distributions.Categorical(logits=0 * discrete).sample()
            rand_action = F.one_hot(indices, discrete.shape[-1])
            probs = ptu.rand(discrete.shape[:1])
            # epsilon greedy
            discrete = torch.where(
                probs.reshape(-1, 1) < expl_amount,
                rand_action.int(),
                discrete.int(),
            )
            continuous = Normal(continuous, expl_amount).rsample()
            if self.use_tanh_normal:
                continuous = torch.clamp(continuous, -1, 1)
            assert (discrete.sum(dim=1) == ptu.ones(discrete.shape[0])).all()
            action = torch.cat((discrete, continuous), -1)
        else:
            action = Normal(action.float(), expl_amount).rsample()
            if self.use_tanh_normal:
                action = torch.clamp(action, -1, 1)
        return action


class ConditionalActorModel(torch.nn.Module):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        env,
        discrete_continuous_dist=True,
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
        self.discrete_actor = Mlp(
            hidden_sizes,
            input_size=obs_dim,
            output_size=discrete_action_dim,
            hidden_activation=hidden_activation,
            hidden_init=torch.nn.init.xavier_uniform_,
            **kwargs
        )
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
        self.discrete_continuous_dist = discrete_continuous_dist

    def forward(self, input):
        h = input
        discrete_logits = self.discrete_actor(h)
        dist1 = OneHotDist(logits=discrete_logits)

        dist = SplitConditionalDist(
            dist1,
            h,
            self.continuous_actor,
            self.use_tanh_normal,
            self.continuous_action_dim,
            self._mean_scale,
            self.raw_init_std,
            self._min_std,
            self.use_per_primitive_actor,
            self.env,
        )
        return dist

    def compute_exploration_action(self, action, expl_amount):
        discrete, continuous = (
            action[:, : self.discrete_action_dim],
            action[:, self.discrete_action_dim :],
        )
        indices = torch.distributions.Categorical(logits=0 * discrete).sample()
        rand_action = F.one_hot(indices, discrete.shape[-1])
        probs = ptu.rand(discrete.shape[:1])
        # epsilon greedy
        discrete = torch.where(
            probs.reshape(-1, 1) < expl_amount,
            rand_action.int(),
            discrete.int(),
        )
        continuous = Normal(continuous, expl_amount).rsample()
        if self.use_tanh_normal:
            continuous = torch.clamp(continuous, -1, 1)
        assert (discrete.sum(dim=1) == ptu.ones(discrete.shape[0])).all()
        action = torch.cat((discrete, continuous), -1)
        return action

    def compute_continuous_exploration_action(self, continuous, expl_amount):
        continuous = Normal(continuous, expl_amount).rsample()
        if self.use_tanh_normal:
            continuous = torch.clamp(continuous, -1, 1)
        return continuous


# "atanh", "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class TanhBijector(Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True

    @property
    def sign(self):
        return 1.0

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (torch.log(ptu.tensor(2.0)) - x - F.softplus(-2.0 * x))


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

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
        indices = (
            torch.argmax(logprob, dim=0)
            .reshape(1, batch_size, 1)
            .expand(1, batch_size, feature_size)
        )
        return torch.gather(sample, 0, indices).squeeze(0)

    def log_prob(self, actions):
        return self._dist.log_prob(actions)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def rsample(self):
        return self._dist.rsample()


class OneHotDist(OneHotCategorical):
    def mode(self):
        return self._one_hot(torch.argmax(self.probs, dim=-1))

    def rsample(self, sample_shape=torch.Size()):
        sample = super().sample(sample_shape)
        probs = self.probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - (probs).detach()  # straight through estimator
        return sample

    def _one_hot(self, indices):
        return F.one_hot(indices, self._categorical._num_events)


class SplitDist:
    def __init__(self, dist1, dist2):
        self._dist1 = dist1
        self._dist2 = dist2

    def rsample(self):
        return torch.cat((self._dist1.rsample(), self._dist2.rsample()), -1)

    def mode(self):
        return torch.cat((self._dist1.mode().float(), self._dist2.mode().float()), -1)

    def entropy(self):
        return self._dist1.entropy() + self._dist2.entropy()

    def log_prob(self, actions):
        return self._dist1.log_prob(actions[:, :13]) + self._dist2.log_prob(
            actions[:, 13:]
        )


class SplitConditionalDist:
    def __init__(
        self,
        dist1,
        h,
        continuous_actor,
        use_tanh_normal,
        continuous_action_dim,
        mean_scale,
        raw_init_std,
        min_std,
        use_per_primitive_actor,
        env,
    ):
        self._dist1 = dist1
        self.h = h
        self.continuous_actor = continuous_actor
        self.use_tanh_normal = use_tanh_normal
        self.continuous_action_dim = continuous_action_dim
        self._mean_scale = mean_scale
        self.raw_init_std = raw_init_std
        self._min_std = min_std
        self.use_per_primitive_actor = use_per_primitive_actor
        self.env = env

    def compute_continuous_dist(self, discrete_action):
        if self.use_per_primitive_actor:
            primitive_indices = torch.argmax(discrete_action, dim=1)
            mean = ptu.zeros((primitive_indices.shape[0], self.continuous_action_dim))
            std = ptu.zeros((primitive_indices.shape[0], self.continuous_action_dim))
            for k, v in self.env.primitive_name_to_action_idx.items():
                if type(v) is int:
                    v = [v]
                if len(v) < 1:
                    continue
                output = self.continuous_actor[k](self.h)
                mean = mean.type(output.dtype)
                std = std.type(output.dtype)
                mean_, std_ = output.split(len(v), -1)
                mean[:, v] = mean_
                std[:, v] = std_
        else:
            h = torch.cat((self.h, discrete_action), dim=1)
            cont_output = self.continuous_actor(h)
            mean, std = cont_output.split(self.continuous_action_dim, -1)
        if self.use_tanh_normal:
            mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
        std = F.softplus(std + self.raw_init_std) + self._min_std

        dist2 = Normal(mean, std)
        if self.use_tanh_normal:
            dist2 = TransformedDistribution(dist2, TanhBijector())
        dist2 = Independent(dist2, 1)
        dist2 = SampleDist(dist2)
        return dist2

    def rsample(self):
        discrete_action = self._dist1.rsample()
        dist2 = self.compute_continuous_dist(discrete_action)
        return torch.cat((discrete_action, dist2.rsample()), -1)

    def rsample_and_log_prob(self):
        discrete_action = self._dist1.rsample()
        dist2 = self.compute_continuous_dist(discrete_action)
        continuous_action = dist2.rsample()
        action = torch.cat((discrete_action, continuous_action), -1)
        log_prob = self._dist1.log_prob(discrete_action) + dist2.log_prob(
            continuous_action
        )
        return action, log_prob

    def mode(self):
        discrete_mode = self._dist1.mode().float()
        dist2 = self.compute_continuous_dist(discrete_mode)
        return torch.cat((discrete_mode, dist2.mode().float()), -1)

    def entropy(self):
        discrete_entropy = self._dist1.entropy()
        continuous_entropy = ptu.zeros_like(discrete_entropy)
        for i in range(self.env.num_primitives):
            discrete_actions = F.one_hot(
                ptu.ones_like(discrete_entropy, dtype=torch.int64) * i,
                self._dist1._categorical._num_events,
            )
            dist2 = self.compute_continuous_dist(discrete_actions)
            # continuous_entropy += dist2.entropy() * self._dist1.log_prob(
            #     discrete_actions
            # )
            continuous_entropy += (
                dist2.entropy() * self._dist1.log_prob(discrete_actions).detach()
            )  # todo: not sure if it is okay to detach this (i think it should be fine)
        return discrete_entropy + continuous_entropy

    def log_prob(self, actions):
        discrete_actions = actions[:, : self.env.num_primitives]
        dist2 = self.compute_continuous_dist(discrete_actions)
        return self._dist1.log_prob(discrete_actions) + dist2.log_prob(
            actions[:, self.env.num_primitives :]
        )

    def log_prob_given_continuous_dist(self, actions, dist2):
        discrete_actions = actions[:, : self.env.num_primitives]
        return self._dist1.log_prob(discrete_actions) + dist2.log_prob(
            actions[:, self.env.num_primitives :]
        )


class SafeTruncatedNormal(TruncatedNormal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale, low, high)
        self._clip = clip
        self._mult = mult

    def sample(self, *args, **kwargs):
        event = super().sample(*args, **kwargs)
        if self._clip:
            clipped = torch.clamp(event, self.a + self._clip, self.b - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event
