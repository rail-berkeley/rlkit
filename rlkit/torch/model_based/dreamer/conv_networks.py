import numpy as np
import torch
from torch import nn as nn

from rlkit.pythonplusplus import identity


class DepthWiseSeparableConv2D(nn.Module):
    def __init__(
        self,
        input_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
    ):
        super(DepthWiseSeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=input_channels,
        )
        self.pointwise = nn.Conv2d(
            input_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=padding,
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DepthWiseSeparableConvTranspose2D(nn.Module):
    def __init__(
        self,
        input_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
    ):
        super(DepthWiseSeparableConvTranspose2D, self).__init__()
        self.depthwise = nn.ConvTranspose2d(
            input_channels,
            input_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=input_channels,
        )
        self.pointwise = nn.ConvTranspose2d(
            input_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=padding,
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class CNN(nn.Module):
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
        use_depth_wise_separable_conv=False,
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
            if use_depth_wise_separable_conv:
                conv = DepthWiseSeparableConv2D(
                    input_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                )
                hidden_init(conv.depthwise.weight)
                conv.depthwise.bias.data.fill_(0)

                hidden_init(conv.pointwise.weight)
                conv.pointwise.bias.data.fill_(0)
            else:
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

    def forward(self, input):
        conv_input = input.narrow(
            start=0, length=self.conv_input_length, dim=1
        ).contiguous()
        # need to reshape from batch of flattened images into (channels, h, w)
        h = conv_input.view(
            conv_input.shape[0],
            self.input_channels,
            self.input_height,
            self.input_width,
        )

        for layer in self.conv_layers:
            h = layer(h)
            h = self.hidden_activation(h)
        # flatten channels for fc layers
        output = h.view(h.size(0), -1)
        return output


class DCNN(nn.Module):
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
        use_depth_wise_separable_conv=False,
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
            if use_depth_wise_separable_conv:
                deconv = DepthWiseSeparableConvTranspose2D(
                    deconv_input_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                )
                hidden_init(deconv.depthwise.weight)
                deconv.depthwise.bias.data.fill_(0)

                hidden_init(deconv.pointwise.weight)
                deconv.pointwise.bias.data.fill_(0)
            else:
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
        if use_depth_wise_separable_conv:
            self.deconv_output = DepthWiseSeparableConvTranspose2D(
                deconv_input_channels,
                deconv_output_channels,
                deconv_output_kernel_size,
                stride=deconv_output_strides,
                padding=0,
            )
            hidden_init(self.deconv_output.depthwise.weight)
            self.deconv_output.depthwise.bias.data.fill_(0)

            hidden_init(self.deconv_output.pointwise.weight)
            self.deconv_output.pointwise.bias.data.fill_(0)
        else:
            self.deconv_output = nn.ConvTranspose2d(
                deconv_input_channels,
                deconv_output_channels,
                deconv_output_kernel_size,
                stride=deconv_output_strides,
            )
            hidden_init(self.deconv_output.weight)
            self.deconv_output.bias.data.fill_(0)

    def forward(self, input):
        h = self.hidden_activation(self.last_fc(input))
        h = h.view(
            -1,
            self.deconv_input_channels,
            self.deconv_input_width,
            self.deconv_input_height,
        )
        h = self.apply_forward(h, self.deconv_layers)
        output = self.deconv_output(h)
        return output

    def apply_forward(self, input, hidden_layers):
        h = input
        for layer in hidden_layers:
            h = layer(h)
            h = self.hidden_activation(h)
        return h
