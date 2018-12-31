import torch
from torch.autograd import Variable

import rlkit.torch.pytorch_util as ptu
try:
    from torch.distributions import Distribution, Normal
except ImportError:
    print("You should use a PyTorch version that has torch.distributions.")
    print("See docker/rlkit/rlkit-env.yml")
    import math
    from numbers import Number
    class Distribution(object):
        r"""
        Distribution is the abstract base class for probability distributions.
        """

        def sample(self):
            """
            Generates a single sample or single batch of samples if the distribution
            parameters are batched.
            """
            raise NotImplementedError

        def sample_n(self, n):
            """
            Generates n samples or n batches of samples if the distribution parameters
            are batched.
            """
            raise NotImplementedError

        def log_prob(self, value):
            """
            Returns the log of the probability density/mass function evaluated at
            `value`.

            Args:
                value (Tensor or Variable):
            """
            raise NotImplementedError

    class Normal(Distribution):
        """
        Creates a normal (also called Gaussian) distribution parameterized by
        `mean` and `std`.

        Example::

            >>> m = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
            >>> m.sample()  # normally distributed with mean=0 and stddev=1
             0.1046
            [torch.FloatTensor of size 1]

        Args:
            mean (float or Tensor or Variable): mean of the distribution
            std (float or Tensor or Variable): standard deviation of the distribution
        """

        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def sample(self):
            return torch.normal(self.mean, self.std)

        def sample_n(self, n):
            # cleanly expand float or Tensor or Variable parameters
            def expand(v):
                if isinstance(v, Number):
                    return torch.Tensor([v]).expand(n, 1)
                else:
                    return v.expand(n, *v.size())
            return torch.normal(expand(self.mean), expand(self.std))

        def log_prob(self, value):
            # compute the variance
            var = (self.std ** 2)
            log_std = math.log(self.std) if isinstance(self.std, Number) else self.std.log()
            return -((value - self.mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))


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
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        z = self.normal.sample()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        z = (
            self.normal_mean +
            self.normal_std *
            Variable(Normal(
                ptu.zeros(self.normal_mean.size()),
                ptu.ones(self.normal_std.size())
            ).sample())
        )
        # z.requires_grad_()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)
