from rlkit.torch.core import PyTorchModule
from rlkit.torch.model_based.dreamer.mlp import Mlp
import torch
import torch.nn.functional as F


class OneStepEnsembleModel(PyTorchModule):
    def __init__(
        self,
        deterministic_state_size=400,
        model_act=F.elu,
        num_models=5,
        hidden_size=32 * 32,
        num_layers=2,
    ):
        super().__init__()
        self.ensemble = [
            Mlp(
                hidden_sizes=[hidden_size] * num_layers,
                input_size=deterministic_state_size,
                output_size=deterministic_state_size * 2,
                hidden_activation=model_act,
                hidden_init=torch.nn.init.xavier_uniform_,
            )
            for i in range(num_models)
        ]
        self.num_models = num_models
        self.model_act = model_act
        self.modules = self.ensemble
        self.deterministic_state_size = deterministic_state_size

    def forward(self, input):
        return [
            self.get_dist(self.ensemble[i](input).split(self.deterministic_state_size))
            for i in range(self.num_models)
        ]

    def get_dist(self, mean, std, dims=1):
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, std), dims
        )
