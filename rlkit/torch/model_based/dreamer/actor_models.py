import torch
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.model_based.dreamer.mlp import Mlp
from rlkit.torch.model_based.dreamer.truncated_normal import TruncatedNormal


class ActorModel(Mlp):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        discrete_continuous_dist=False,
        discrete_action_dim=0,
        continuous_action_dim=0,
        hidden_activation=F.elu,
        min_std=1e-4,
        init_std=5.0,
        mean_scale=5.0,
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

            action_mean = self._mean_scale * torch.tanh(
                continuous_action_mean / self._mean_scale
            )
            action_std = (
                F.softplus(continuous_action_std + raw_init_std) + self._min_std
            )

            dist2 = Normal(action_mean, action_std)
            dist2 = TransformedDistribution(dist2, TanhBijector())
            dist2 = torch.distributions.Independent(dist2, 1)
            dist2 = SampleDist(dist2)
            dist = SplitDist(dist1, dist2)
        else:
            action_mean, action_std_dev = last.split(self.continuous_action_dim, -1)
            action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
            action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std

            dist = Normal(action_mean, action_std)
            dist = TransformedDistribution(dist, TanhBijector())
            dist = torch.distributions.Independent(dist, 1)
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


class OneHotDist(torch.distributions.one_hot_categorical.OneHotCategorical):
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
