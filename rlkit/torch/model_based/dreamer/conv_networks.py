import numpy as np
import torch
from torch import jit
from torch import nn as nn

from rlkit.pythonplusplus import identity
from rlkit.torch.model_based.dreamer.mlp import Mlp


class CNN(jit.ScriptModule):
    def __init__(
        self,
        input_width,
        input_height,
        input_channels,
        kernel_sizes,
        n_channels,
        strides,
        paddings,
        hidden_sizes=None,
        hidden_init=nn.init.xavier_uniform_,
        hidden_activation=nn.ReLU(),
        output_activation=identity,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == len(n_channels) == len(strides) == len(paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.conv_input_length = (
            self.input_width * self.input_height * self.input_channels
        )

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()

        for out_channels, kernel_size, stride, padding in zip(
            n_channels, kernel_sizes, strides, paddings
        ):
            conv = nn.Conv2d(
                input_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            )
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels
        self.to(memory_format=torch.channels_last)

    @jit.script_method
    def forward(self, input):
        conv_input = input.narrow(
            start=0, length=self.conv_input_length, dim=1
        ).contiguous()
        h = conv_input.view(
            conv_input.shape[0],
            self.input_channels,
            self.input_height,
            self.input_width,
        ).to(device="cuda", memory_format=torch.channels_last, dtype=torch.float16)

        for layer in self.conv_layers:
            h = layer(h)
            h = self.hidden_activation(h)
        output = h.reshape(h.size(0), -1)
        return output


class DCNN(jit.ScriptModule):
    def __init__(
        self,
        fc_input_size,
        hidden_sizes,
        deconv_input_width,
        deconv_input_height,
        deconv_input_channels,
        deconv_output_kernel_size,
        deconv_output_strides,
        deconv_output_channels,
        kernel_sizes,
        n_channels,
        strides,
        paddings,
        batch_norm_deconv=False,
        batch_norm_fc=False,
        hidden_init=nn.init.xavier_uniform_,
        hidden_activation=nn.ReLU(),
        output_activation=identity,
    ):
        assert len(kernel_sizes) == len(n_channels) == len(strides) == len(paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

        self.deconv_input_width = deconv_input_width
        self.deconv_input_height = deconv_input_height
        self.deconv_input_channels = deconv_input_channels
        deconv_input_size = (
            self.deconv_input_channels
            * self.deconv_input_height
            * self.deconv_input_width
        )
        self.batch_norm_deconv = batch_norm_deconv
        self.batch_norm_fc = batch_norm_fc

        self.deconv_layers = nn.ModuleList()

        self.last_fc = nn.Linear(fc_input_size, deconv_input_size)
        hidden_init(self.last_fc.weight)
        self.last_fc.bias.data.fill_(0)

        for out_channels, kernel_size, stride, padding in zip(
            n_channels, kernel_sizes, strides, paddings
        ):
            deconv = nn.ConvTranspose2d(
                deconv_input_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            )
            hidden_init(deconv.weight)
            deconv.bias.data.fill_(0)

            deconv_layer = deconv
            self.deconv_layers.append(deconv_layer)
            deconv_input_channels = out_channels
        self.deconv_output = nn.ConvTranspose2d(
            deconv_input_channels,
            deconv_output_channels,
            deconv_output_kernel_size,
            stride=deconv_output_strides,
        )
        hidden_init(self.deconv_output.weight)
        self.deconv_output.bias.data.fill_(0)
        self.to(memory_format=torch.channels_last)

    @jit.script_method
    def forward(self, input):
        h = self.hidden_activation(self.last_fc(input))
        h = h.view(
            -1,
            self.deconv_input_channels,
            self.deconv_input_width,
            self.deconv_input_height,
        ).to(device="cuda", memory_format=torch.channels_last, dtype=torch.float16)
        for layer in self.deconv_layers:
            h = layer(h)
            h = self.hidden_activation(h)
        output = self.deconv_output(h)
        return output
