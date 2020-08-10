import torch
from torch import nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule, eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.networks import LayerNorm
from rlkit.torch.pytorch_util import activation_from_string


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
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
        self.output_size = output_size
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

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class MultiHeadedMlp(Mlp):
    """
                   .-> linear head 0
                  /
    input --> MLP ---> linear head 1
                  \
                   .-> linear head 2
    """
    def __init__(
            self,
            hidden_sizes,
            output_sizes,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activations=None,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            output_size=sum(output_sizes),
            input_size=input_size,
            init_w=init_w,
            hidden_activation=hidden_activation,
            hidden_init=hidden_init,
            b_init_value=b_init_value,
            layer_norm=layer_norm,
            layer_norm_kwargs=layer_norm_kwargs,
        )
        self._splitter = SplitIntoManyHeads(
            output_sizes,
            output_activations,
        )

    def forward(self, input):
        flat_outputs = super().forward(input)
        return self._splitter(flat_outputs)


class ConcatMultiHeadedMlp(MultiHeadedMlp):
    """
    Concatenate inputs along dimension and then pass through MultiHeadedMlp.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)


class ConcatMlp(Mlp):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class MlpQf(ConcatMlp):
    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            action_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer

    def forward(self, obs, actions, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        if self.action_normalizer:
            actions = self.action_normalizer.normalize(actions)
        return super().forward(obs, actions, **kwargs)


class MlpQfWithObsProcessor(Mlp):
    def __init__(self, obs_processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_processor = obs_processor

    def forward(self, obs, actions, **kwargs):
        h = self.obs_processor(obs)
        flat_inputs = torch.cat((h, actions), dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpGoalQfWithObsProcessor(Mlp):
    def __init__(self, obs_processor, obs_dim,
                 backprop_into_obs_preprocessor=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_processor = obs_processor
        self.backprop_into_obs_preprocessor = backprop_into_obs_preprocessor
        self.obs_dim = obs_dim

    def forward(self, obs, actions, **kwargs):
        h_s = self.obs_processor(obs[:, :self.obs_dim])
        h_g = self.obs_processor(obs[:, self.obs_dim:])
        if not self.backprop_into_obs_preprocessor:
            h_s = h_s.detach()
            h_g = h_g.detach()
        flat_inputs = torch.cat((h_s, h_g, actions), dim=1)
        return super().forward(flat_inputs, **kwargs)


class SplitIntoManyHeads(nn.Module):
    """
           .-> head 0
          /
    input ---> head 1
          \
           '-> head 2
    """
    def __init__(
            self,
            output_sizes,
            output_activations=None,
    ):
        super().__init__()
        if output_activations is None:
            output_activations = ['identity' for _ in output_sizes]
        else:
            if len(output_activations) != len(output_sizes):
                raise ValueError("output_activation and output_sizes must have "
                                 "the same length")

        self._output_narrow_params = []
        self._output_activations = []
        for output_activation in output_activations:
            if isinstance(output_activation, str):
                output_activation = activation_from_string(output_activation)
            self._output_activations.append(output_activation)
        start_idx = 0
        for output_size in output_sizes:
            self._output_narrow_params.append((start_idx, output_size))
            start_idx = start_idx + output_size

    def forward(self, flat_outputs):
        pre_activation_outputs = tuple(
            flat_outputs.narrow(1, start, length)
            for start, length in self._output_narrow_params
        )
        outputs = tuple(
            activation(x)
            for activation, x in zip(
                self._output_activations, pre_activation_outputs
            )
        )
        return outputs


class ParallelMlp(nn.Module):
    """
    Efficient implementation of multiple MLPs with identical architectures.

           .-> mlp 0
          /
    input ---> mlp 1
          \
           '-> mlp 2

    See https://discuss.pytorch.org/t/parallel-execution-of-modules-in-nn-modulelist/43940/7
    for details

    The last dimension of the output corresponds to the MLP index.
    """
    def __init__(
            self,
            num_heads,
            input_size,
            output_size_per_mlp,
            hidden_sizes,
            hidden_activation='relu',
            output_activation='identity',
            input_is_already_expanded=False,
    ):
        super().__init__()

        def create_layers():
            layers = []
            input_dim = input_size
            for i, hidden_size in enumerate(hidden_sizes):
                fc = nn.Conv1d(
                    in_channels=input_dim * num_heads,
                    out_channels=hidden_size * num_heads,
                    kernel_size=1,
                    groups=num_heads,
                )
                layers.append(fc)
                if isinstance(hidden_activation, str):
                    activation = activation_from_string(hidden_activation)
                else:
                    activation = hidden_activation
                layers.append(activation)
                input_dim = hidden_size

            last_fc = nn.Conv1d(
                in_channels=input_dim * num_heads,
                out_channels=output_size_per_mlp * num_heads,
                kernel_size=1,
                groups=num_heads,
            )
            layers.append(last_fc)
            if output_activation != 'identity':
                if isinstance(output_activation, str):
                    activation = activation_from_string(output_activation)
                else:
                    activation = output_activation
                layers.append(activation)
            return layers

        self.network = nn.Sequential(*create_layers())
        self.num_heads = num_heads
        self.input_is_already_expanded = input_is_already_expanded

    def forward(self, x):
        if not self.input_is_already_expanded:
            x = x.repeat(1, self.num_heads).unsqueeze(-1)
        flat = self.network(x)
        batch_size = x.shape[0]
        return flat.view(batch_size, -1, self.num_heads)
