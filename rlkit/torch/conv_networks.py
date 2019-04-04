import torch
from torch import nn as nn

from rlkit.pythonplusplus import identity

import numpy as np


class CNN(nn.Module):
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes=None,
            added_fc_input_size=0,
            batch_norm_conv=False,
            batch_norm_fc=False,
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.batch_norm_conv = batch_norm_conv
        self.batch_norm_fc = batch_norm_fc
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        for out_channels, kernel_size, stride, padding in \
                zip(n_channels, kernel_sizes, strides, paddings):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

        # find output dim of conv_layers by trial and add normalization conv layers
        test_mat = torch.zeros(1, self.input_channels, self.input_width,
                               self.input_height)  # initially the model is on CPU (caller should then move it to GPU if
        for conv_layer in self.conv_layers:
            test_mat = conv_layer(test_mat)
            self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))

        fc_input_size = int(np.prod(test_mat.shape))
        # used only for injecting input directly into fc layers
        fc_input_size += added_fc_input_size

        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)

            norm_layer = nn.BatchNorm1d(hidden_size)
            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)

            self.fc_layers.append(fc_layer)
            self.fc_norm_layers.append(norm_layer)
            fc_input_size = hidden_size

        self.last_fc = nn.Linear(fc_input_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input):
        fc_input = (self.added_fc_input_size != 0)

        conv_input = input.narrow(start=0,
                                  length=self.conv_input_length,
                                  dim=1).contiguous()
        if fc_input:
            extra_fc_input = input.narrow(start=self.conv_input_length,
                                          length=self.added_fc_input_size,
                                          dim=1)
        # need to reshape from batch of flattened images into (channsls, w, h)
        h = conv_input.view(conv_input.shape[0],
                            self.input_channels,
                            self.input_height,
                            self.input_width)

        h = self.apply_forward(h, self.conv_layers, self.conv_norm_layers,
                               use_batch_norm=self.batch_norm_conv)
        # flatten channels for fc layers
        h = h.view(h.size(0), -1)
        if fc_input:
            h = torch.cat((h, extra_fc_input), dim=1)
        h = self.apply_forward(h, self.fc_layers, self.fc_norm_layers,
                               use_batch_norm=self.batch_norm_fc)

        output = self.output_activation(self.last_fc(h))
        return output

    def apply_forward(self, input, hidden_layers, norm_layers,
                      use_batch_norm=False):
        h = input
        for layer, norm_layer in zip(hidden_layers, norm_layers):
            h = layer(h)
            if use_batch_norm:
                h = norm_layer(h)
            h = self.hidden_activation(h)
        return h


class TwoHeadDCNN(nn.Module):
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
            init_w=1e-3,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
    ):
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

        self.deconv_input_width = deconv_input_width
        self.deconv_input_height = deconv_input_height
        self.deconv_input_channels = deconv_input_channels
        deconv_input_size = self.deconv_input_channels * self.deconv_input_height * self.deconv_input_width
        self.batch_norm_deconv = batch_norm_deconv
        self.batch_norm_fc = batch_norm_fc

        self.deconv_layers = nn.ModuleList()
        self.deconv_norm_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)

            norm_layer = nn.BatchNorm1d(hidden_size)
            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)

            self.fc_layers.append(fc_layer)
            self.fc_norm_layers.append(norm_layer)
            fc_input_size = hidden_size

        self.last_fc = nn.Linear(fc_input_size, deconv_input_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        for out_channels, kernel_size, stride, padding in \
                zip(n_channels, kernel_sizes, strides, paddings):
            deconv = nn.ConvTranspose2d(deconv_input_channels,
                                        out_channels,
                                        kernel_size,
                                        stride=stride,
                                        padding=padding)
            hidden_init(deconv.weight)
            deconv.bias.data.fill_(0)

            deconv_layer = deconv
            self.deconv_layers.append(deconv_layer)
            deconv_input_channels = out_channels

        test_mat = torch.zeros(1, self.deconv_input_channels,
                               self.deconv_input_width,
                               self.deconv_input_height)  # initially the model is on CPU (caller should then move it to GPU if
        for deconv_layer in self.deconv_layers:
            test_mat = deconv_layer(test_mat)
            self.deconv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))

        self.first_deconv_output = nn.ConvTranspose2d(
            deconv_input_channels,
            deconv_output_channels,
            deconv_output_kernel_size,
            stride=deconv_output_strides,
        )
        hidden_init(self.first_deconv_output.weight)
        self.first_deconv_output.bias.data.fill_(0)

        self.second_deconv_output = nn.ConvTranspose2d(
            deconv_input_channels,
            deconv_output_channels,
            deconv_output_kernel_size,
            stride=deconv_output_strides,
        )
        hidden_init(self.second_deconv_output.weight)
        self.second_deconv_output.bias.data.fill_(0)

    def forward(self, input):
        h = self.apply_forward(input, self.fc_layers, self.fc_norm_layers,
                               use_batch_norm=self.batch_norm_fc)
        h = self.hidden_activation(self.last_fc(h))
        h = h.view(-1, self.deconv_input_channels, self.deconv_input_width,
                   self.deconv_input_height)
        h = self.apply_forward(h, self.deconv_layers, self.deconv_norm_layers,
                               use_batch_norm=self.batch_norm_deconv)
        first_output = self.output_activation(self.first_deconv_output(h))
        second_output = self.output_activation(self.second_deconv_output(h))
        return first_output, second_output

    def apply_forward(self, input, hidden_layers, norm_layers,
                      use_batch_norm=False):
        h = input
        for layer, norm_layer in zip(hidden_layers, norm_layers):
            h = layer(h)
            if use_batch_norm:
                h = norm_layer(h)
            h = self.hidden_activation(h)
        return h


class DCNN(TwoHeadDCNN):
    def forward(self, x):
        return super().forward(x)[0]
