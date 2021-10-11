"""
Add custom distributions in addition to th existing ones
"""
import torch
from torch.distributions import Categorical, OneHotCategorical, kl_divergence
from torch.distributions import Normal as TorchNormal
from torch.distributions import Beta as TorchBeta
from torch.distributions import Distribution as TorchDistribution
from torch.distributions import Bernoulli as TorchBernoulli
from torch.distributions import Independent as TorchIndependent
from torch.distributions.utils import _sum_rightmost
from rlkit.core.eval_util import create_stats_ordered_dict
import rlkit.torch.pytorch_util as ptu
import numpy as np
from collections import OrderedDict

class Distribution(TorchDistribution):
    def sample_and_logprob(self):
        s = self.sample()
        log_p = self.log_prob(s)
        return s, log_p

    def rsample_and_logprob(self):
        s = self.rsample()
        log_p = self.log_prob(s)
        return s, log_p

    def mle_estimate(self):
        return self.mean

    def get_diagnostics(self):
        return {}


class TorchDistributionWrapper(Distribution):
    def __init__(self, distribution: TorchDistribution):
        self.distribution = distribution

    @property
    def batch_shape(self):
        return self.distribution.batch_shape

    @property
    def event_shape(self):
        return self.distribution.event_shape

    @property
    def arg_constraints(self):
        return self.distribution.arg_constraints

    @property
    def support(self):
        return self.distribution.support

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def variance(self):
        return self.distribution.variance

    @property
    def stddev(self):
        return self.distribution.stddev

    def sample(self, sample_size=torch.Size()):
        return self.distribution.sample(sample_shape=sample_size)

    def rsample(self, sample_size=torch.Size()):
        return self.distribution.rsample(sample_shape=sample_size)

    def log_prob(self, value):
        return self.distribution.log_prob(value)

    def cdf(self, value):
        return self.distribution.cdf(value)

    def icdf(self, value):
        return self.distribution.icdf(value)

    def enumerate_support(self, expand=True):
        return self.distribution.enumerate_support(expand=expand)

    def entropy(self):
        return self.distribution.entropy()

    def perplexity(self):
        return self.distribution.perplexity()

    def __repr__(self):
        return 'Wrapped ' + self.distribution.__repr__()


class Delta(Distribution):
    """A deterministic distribution"""
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value.detach()

    def rsample(self):
        return self.value

    @property
    def mean(self):
        return self.value

    @property
    def variance(self):
        return 0

    @property
    def entropy(self):
        return 0


class Bernoulli(Distribution, TorchBernoulli):
    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict(
            'probability',
            ptu.get_numpy(self.probs),
        ))
        return stats


class Independent(Distribution, TorchIndependent):
    def get_diagnostics(self):
        return self.base_dist.get_diagnostics()


class Beta(Distribution, TorchBeta):
    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict(
            'alpha',
            ptu.get_numpy(self.concentration0),
        ))
        stats.update(create_stats_ordered_dict(
            'beta',
            ptu.get_numpy(self.concentration1),
        ))
        stats.update(create_stats_ordered_dict(
            'entropy',
            ptu.get_numpy(self.entropy()),
        ))
        return stats


class MultivariateDiagonalNormal(TorchDistributionWrapper):
    from torch.distributions import constraints
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}

    def __init__(self, loc, scale_diag, reinterpreted_batch_ndims=1):
        dist = Independent(TorchNormal(loc, scale_diag),
                           reinterpreted_batch_ndims=reinterpreted_batch_ndims)
        super().__init__(dist)

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict(
            'mean',
            ptu.get_numpy(self.mean),
            # exclude_max_min=True,
        ))
        stats.update(create_stats_ordered_dict(
            'std',
            ptu.get_numpy(self.distribution.stddev),
        ))
        return stats

    def __repr__(self):
        return self.distribution.base_dist.__repr__()


@torch.distributions.kl.register_kl(TorchDistributionWrapper,
                                    TorchDistributionWrapper)
def _kl_mv_diag_normal_mv_diag_normal(p, q):
    return kl_divergence(p.distribution, q.distribution)

# Independent RV KL handling - https://github.com/pytorch/pytorch/issues/13545

@torch.distributions.kl.register_kl(TorchIndependent, TorchIndependent)
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)

class GaussianMixture(Distribution):
    def __init__(self, normal_means, normal_stds, weights):
        self.num_gaussians = weights.shape[1]
        self.normal_means = normal_means
        self.normal_stds = normal_stds
        self.normal = MultivariateDiagonalNormal(normal_means, normal_stds)
        self.normals = [MultivariateDiagonalNormal(normal_means[:, :, i], normal_stds[:, :, i]) for i in range(self.num_gaussians)]
        self.weights = weights
        self.categorical = OneHotCategorical(self.weights[:, :, 0])

    def log_prob(self, value, ):
        log_p = [self.normals[i].log_prob(value) for i in range(self.num_gaussians)]
        log_p = torch.stack(log_p, -1)
        log_p = log_p.sum(dim=1)
        log_weights = torch.log(self.weights[:, :, 0])
        lp = log_weights + log_p
        m = lp.max(dim=1)[0]  # log-sum-exp numerical stability trick
        log_p_mixture = m + torch.log(torch.exp(lp - m).sum(dim=1))
        return log_p_mixture

    def sample(self):
        z = self.normal.sample().detach()
        c = self.categorical.sample()[:, :, None]
        s = torch.matmul(z, c)
        return torch.squeeze(s, 2)

    def rsample(self):
        z = (
                self.normal_means +
                self.normal_stds *
                MultivariateDiagonalNormal(
                    ptu.zeros(self.normal_means.size()),
                    ptu.ones(self.normal_stds.size())
                ).sample()
        )
        z.requires_grad_()
        c = self.categorical.sample()[:, :, None]
        s = torch.matmul(z, c)
        return torch.squeeze(s, 2)

    def mle_estimate(self):
        """Return the mean of the most likely component.

        This often computes the mode of the distribution, but not always.
        """
        c = ptu.zeros(self.weights.shape[:2])
        ind = torch.argmax(self.weights, dim=1) # [:, 0]
        c.scatter_(1, ind, 1)
        s = torch.matmul(self.normal_means, c[:, :, None])
        return torch.squeeze(s, 2)

    def __repr__(self):
        s = "GaussianMixture(normal_means=%s, normal_stds=%s, weights=%s)"
        return s % (self.normal_means, self.normal_stds, self.weights)


epsilon = 0.001


class GaussianMixtureFull(Distribution):
    def __init__(self, normal_means, normal_stds, weights):
        self.num_gaussians = weights.shape[-1]
        self.normal_means = normal_means
        self.normal_stds = normal_stds
        self.normal = MultivariateDiagonalNormal(normal_means, normal_stds)
        self.normals = [MultivariateDiagonalNormal(normal_means[:, :, i], normal_stds[:, :, i]) for i in range(self.num_gaussians)]
        self.weights = (weights + epsilon) / (1 + epsilon * self.num_gaussians)
        assert (self.weights > 0).all()
        self.categorical = Categorical(self.weights)

    def log_prob(self, value, ):
        log_p = [self.normals[i].log_prob(value) for i in range(self.num_gaussians)]
        log_p = torch.stack(log_p, -1)
        log_weights = torch.log(self.weights)
        lp = log_weights + log_p
        m = lp.max(dim=2, keepdim=True)[0]  # log-sum-exp numerical stability trick
        log_p_mixture = m + torch.log(torch.exp(lp - m).sum(dim=2, keepdim=True))
        raise NotImplementedError("from Vitchyr: idk what the point is of "
                                  "this class, so I didn't both updating "
                                  "this, but log_prob should return something "
                                  "of shape [batch_size] and not [batch_size, "
                                  "1] to be in accordance with the "
                                  "torch.distributions.Distribution "
                                  "interface.")
        return torch.squeeze(log_p_mixture, 2)

    def sample(self):
        z = self.normal.sample().detach()
        c = self.categorical.sample()[:, :, None]
        s = torch.gather(z, dim=2, index=c)
        return s[:, :, 0]

    def rsample(self):
        z = (
                self.normal_means +
                self.normal_stds *
                MultivariateDiagonalNormal(
                    ptu.zeros(self.normal_means.size()),
                    ptu.ones(self.normal_stds.size())
                ).sample()
        )
        z.requires_grad_()
        c = self.categorical.sample()[:, :, None]
        s = torch.gather(z, dim=2, index=c)
        return s[:, :, 0]

    def mle_estimate(self):
        """Return the mean of the most likely component.

        This often computes the mode of the distribution, but not always.
        """
        ind = torch.argmax(self.weights, dim=2)[:, :, None]
        means = torch.gather(self.normal_means, dim=2, index=ind)
        return torch.squeeze(means, 2)

    def __repr__(self):
        s = "GaussianMixture(normal_means=%s, normal_stds=%s, weights=%s)"
        return s % (self.normal_means, self.normal_stds, self.weights)


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = MultivariateDiagonalNormal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def _log_prob_from_pre_tanh(self, pre_tanh_value):
        """
        Adapted from
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73

        This formula is mathematically equivalent to log(1 - tanh(x)^2).

        Derivation:

        log(1 - tanh(x)^2)
         = log(sech(x)^2)
         = 2 * log(sech(x))
         = 2 * log(2e^-x / (e^-2x + 1))
         = 2 * (log(2) - x - log(e^-2x + 1))
         = 2 * (log(2) - x - softplus(-2x))

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        log_prob = self.normal.log_prob(pre_tanh_value)
        correction = - 2. * (
            ptu.from_numpy(np.log([2.]))
            - pre_tanh_value
            - torch.nn.functional.softplus(-2. * pre_tanh_value)
        ).sum(dim=1)
        return log_prob + correction

    def log_prob(self, value, pre_tanh_value=None):
        if pre_tanh_value is None:
            # errors or instability at values near 1
            value = torch.clamp(value, -0.999999, 0.999999)
            pre_tanh_value = torch.log(1+value) / 2 - torch.log(1-value) / 2
        return self._log_prob_from_pre_tanh(pre_tanh_value)

    def rsample_with_pretanh(self):
        z = (
                self.normal_mean +
                self.normal_std *
                MultivariateDiagonalNormal(
                    ptu.zeros(self.normal_mean.size()),
                    ptu.ones(self.normal_std.size())
                ).sample()
        )
        return torch.tanh(z), z

    def sample(self):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        value, pre_tanh_value = self.rsample_with_pretanh()
        return value.detach()

    def rsample(self):
        """
        Sampling in the reparameterization case.
        """
        value, pre_tanh_value = self.rsample_with_pretanh()
        return value

    def sample_and_logprob(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        value, pre_tanh_value = value.detach(), pre_tanh_value.detach()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    def rsample_and_logprob(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    def rsample_logprob_and_pretanh(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p, pre_tanh_value

    @property
    def mean(self):
        return torch.tanh(self.normal_mean)

    @property
    def stddev(self):
        return self.normal_std

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict(
            'mean',
            ptu.get_numpy(self.mean),
        ))
        stats.update(create_stats_ordered_dict(
            'normal/std',
            ptu.get_numpy(self.normal_std)
        ))
        stats.update(create_stats_ordered_dict(
            'normal/log_std',
            ptu.get_numpy(torch.log(self.normal_std)),
        ))
        return stats