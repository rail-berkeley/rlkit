from rlkit.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)
from rlkit.torch.sac.policies.base import (
    TorchStochasticPolicy,
    PolicyFromDistributionGenerator,
    MakeDeterministic,
)

from rlkit.torch.lvm.latent_variable_model import LatentVariableModel


class LVMPolicy(LatentVariableModel, TorchStochasticPolicy):
    """Expects encoder p(z|s) and decoder p(u|s,z)"""

    def forward(self, obs):
        z_dist = self.encoder(obs)
        z = z_dist.sample()
        return self.decoder(obs, z)
