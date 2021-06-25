import torch
import torch.nn.functional as F
from torch import jit

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.model_based.dreamer.mlp import Mlp


class OneStepEnsembleModel(jit.ScriptModule):
    def __init__(
        self,
        action_dim,
        deterministic_state_size,
        stochastic_state_size,
        embedding_size,
        model_act=F.elu,
        num_models=10,
        hidden_size=400,
        num_layers=4,
        inputs="feat",
        targets="stoch",
    ):
        super().__init__()
        self.ensemble = torch.nn.ModuleList()
        self.size = {
            "embed": embedding_size,
            "stoch": stochastic_state_size,
            "deter": deterministic_state_size,
            "feat": stochastic_state_size + deterministic_state_size,
        }
        self.inputs = inputs
        self.targets = targets
        self.input_size = self.size[inputs]
        self.output_size = self.size[targets]
        for i in range(num_models):
            self.ensemble.append(
                Mlp(
                    hidden_sizes=[hidden_size] * num_layers,
                    input_size=self.input_size + action_dim,
                    output_size=self.output_size,
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
