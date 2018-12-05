import torch
from rlkit.torch.core import PyTorchModule
import numpy as np
import abc
from rlkit.torch import pytorch_util as ptu


class VAEBase(PyTorchModule, metaclass=abc.ABCMeta):
    def __init__(
            self,
            representation_size,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.representation_size = representation_size

    @abc.abstractmethod
    def encode(self, input):
        """
        :param input:
        :return: latent_distribution_params
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def rsample(self, latent_distribution_params):
        """

        :param latent_distribution_params:
        :return: latents
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reparameterize(self, latent_distribution_params):
        """

        :param latent_distribution_params:
        :return: latents
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, latents):
        """
        :param latents:
        :return: reconstruction, obs_distribution_params
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def logprob(self, inputs, obs_distribution_params):
        """
        :param inputs:
        :param obs_distribution_params:
        :return: log probability of input under decoder
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def kl_divergence(self, latent_distribution_params):
        """
        :param latent_distribution_params:
        :return: kl div between latent_distribution_params and prior on latent space
        """
        raise NotImplementedError()

    def forward(self, input):
        """
        :param input:
        :return: reconstructed input, obs_distribution_params, latent_distribution_params
        """
        latent_distribution_params = self.encode(input)
        latents = self.reparameterize(latent_distribution_params)
        reconstructions, obs_distribution_params = self.decode(latents)
        return reconstructions, obs_distribution_params, latent_distribution_params


class GaussianLatentVAE(VAEBase):
    def __init__(
            self,
            representation_size,
    ):
        self.save_init_params(locals())
        super().__init__(representation_size)
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)

    def rsample(self, latent_distribution_params):
        mu, logvar = latent_distribution_params
        stds = (0.5 * logvar).exp()
        epsilon = ptu.randn(*mu.size())
        latents = epsilon * stds + mu
        return latents

    def rsample_multiple_latents(self, latent_distribution_params,
                                 num_latents_to_sample=1):
        mu, logvar = latent_distribution_params
        mu = mu.view((mu.size()[0], 1, mu.size()[1]))
        stds = (0.5 * logvar).exp()
        stds = stds.view(stds.size()[0], 1, stds.size()[1])
        epsilon = ptu.randn((mu.size()[0], num_latents_to_sample, mu.size()[1]))
        latents = epsilon * stds + mu
        return latents

    def reparameterize(self, latent_distribution_params):
        if self.training:
            return self.rsample(latent_distribution_params)
        else:
            return latent_distribution_params[0]

    def kl_divergence(self, latent_distribution_params):
        """
        See Appendix B from VAE paper:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114

        Or just look it up.

        0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        Note that sometimes people write log(sigma), but this is the same as
        0.5 * log(sigma^2).

        :param latent_distribution_params:
        :return:
        """
        mu, logvar = latent_distribution_params
        return - 0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        ).mean()

    def __getstate__(self):
        d = super().__getstate__()
        # Add these explicitly in case they were modified
        d["_dist_mu"] = self.dist_mu
        d["_dist_std"] = self.dist_std
        return d

    def __setstate__(self, d):
        super().__setstate__(d)
        self.dist_mu = d["_dist_mu"]
        self.dist_std = d["_dist_std"]
