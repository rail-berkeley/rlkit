import torch
from torch import jit, nn
from torch.nn import functional as F

from rlkit.pythonplusplus import identity
from rlkit.torch.core import PyTorchModule


class Mlp(jit.ScriptModule):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        hidden_activation=F.elu,
        output_activation=identity,
        hidden_init=torch.nn.init.xavier_uniform_,
        b_init_value=0.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.fcs = torch.nn.ModuleList()
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)
        torch.nn.init.xavier_uniform_(self.last_fc.weight)
        self.last_fc.bias.data.fill_(0)

    @jit.script_method
    def forward(self, input):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        return output
