from torch import nn as nn
from torch.nn import functional as F

from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import LayerNorm


class TwoHeadMlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            first_head_size,
            second_head_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.first_head_size = first_head_size
        self.second_head_size = second_head_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.first_head = nn.Linear(in_size, self.first_head_size)
        self.first_head.weight.data.uniform_(-init_w, init_w)

        self.second_head = nn.Linear(in_size, self.second_head_size)
        self.second_head.weight.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.first_head(h)
        first_output = self.output_activation(preactivation)
        preactivation = self.second_head(h)
        second_output = self.output_activation(preactivation)

        return first_output, second_output
