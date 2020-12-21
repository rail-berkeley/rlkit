import torch
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.model_based.dreamer.mlp import Mlp


class OneStepEnsembleModel(PyTorchModule):
    def __init__(
        self,
        action_dim,
        deterministic_state_size,
        embedding_size,
        model_act=F.elu,
        num_models=5,
        hidden_size=32 * 32,
        num_layers=2,
        output_embeddings=True,
    ):
        super().__init__()
        self.ensemble = torch.nn.ModuleList()
        self.output_embeddings = output_embeddings
        if self.output_embeddings:
            output_size = embedding_size
        else:
            output_size = deterministic_state_size
        for i in range(num_models):
            self.ensemble.append(
                Mlp(
                    hidden_sizes=[hidden_size] * num_layers,
                    input_size=deterministic_state_size + action_dim,
                    output_size=output_size,
                    hidden_activation=model_act,
                    hidden_init=torch.nn.init.xavier_uniform_,
                )
            )

        self.num_models = num_models
        self.model_act = model_act

    def forward_ith_model(self, input, i):
        mean = self.ensemble[i](input)
        return self.get_dist(mean, ptu.ones_like(mean))

    def get_dist(self, mean, std, dims=1):
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, std), dims
        )
