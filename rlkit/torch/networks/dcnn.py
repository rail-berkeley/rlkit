import torch
from torch import nn as nn

from rlkit.pythonplusplus import identity
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import activation_from_string


class TwoHeadDCNN(PyTorchModule):
    def __init__(self,
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

                 deconv_normalization_type='none',
                 fc_normalization_type='none',
                 init_w=1e-3,
                 hidden_init=nn.init.xavier_uniform_,
                 hidden_activation=nn.ReLU(),
                 output_activation=identity
                 ):
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        assert deconv_normalization_type in {'none', 'batch', 'layer'}
        assert fc_normalization_type in {'none', 'batch', 'layer'}
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

        self.deconv_input_width = deconv_input_width
        self.deconv_input_height = deconv_input_height
        self.deconv_input_channels = deconv_input_channels
        deconv_input_size = self.deconv_input_channels * self.deconv_input_height * self.deconv_input_width
        self.deconv_normalization_type = deconv_normalization_type
        self.fc_normalization_type = fc_normalization_type

        self.deconv_layers = nn.ModuleList()
        self.deconv_norm_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)

            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)

            self.fc_layers.append(fc_layer)
            if self.fc_normalization_type == 'batch':
                self.fc_norm_layers.append(nn.BatchNorm1d(hidden_size))
            if self.fc_normalization_type == 'layer':
                self.fc_norm_layers.append(nn.LayerNorm(hidden_size))
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

        test_mat = torch.zeros(1, self.deconv_input_channels, self.deconv_input_width,
                               self.deconv_input_height)  # initially the model is on CPU (caller should then move it to GPU if
        for deconv_layer in self.deconv_layers:
            test_mat = deconv_layer(test_mat)
            if self.deconv_normalization_type == 'batch':
                self.deconv_norm_layers.append(
                    nn.BatchNorm2d(test_mat.shape[1])
                )
            if self.deconv_normalization_type == 'layer':
                self.deconv_norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))

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
                               normalization_type=self.fc_normalization_type)
        h = self.hidden_activation(self.last_fc(h))
        h = h.view(-1, self.deconv_input_channels, self.deconv_input_width, self.deconv_input_height)
        h = self.apply_forward(h, self.deconv_layers,
                               self.deconv_norm_layers,
                               normalization_type=self.deconv_normalization_type)
        first_output = self.output_activation(self.first_deconv_output(h))
        second_output = self.output_activation(self.second_deconv_output(h))
        return first_output, second_output

    def apply_forward(self, input, hidden_layers, norm_layers,
                      normalization_type='none'):
        h = input
        for i, layer in enumerate(hidden_layers):
            h = layer(h)
            if normalization_type != 'none':
                h = norm_layers[i](h)
            h = self.hidden_activation(h)
        return h


class DCNN(TwoHeadDCNN):
    def forward(self, input):
        return super().forward(input)[0]


class BasicDCNN(PyTorchModule):
    """Deconvolution neural network."""
    # TODO (maybe?): merge with BasicCNN code
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,

            kernel_sizes,
            n_channels,
            strides,
            paddings,

            normalization_type='none',
            hidden_init=None,
            hidden_activation='relu',
            output_activation=identity,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,
     ):
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        assert normalization_type in {'none', 'batch', 'layer'}
        assert pool_type in {'none', 'max2d'}
        if pool_type == 'max2d':
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)
        super().__init__()

        self.output_activation = output_activation
        if isinstance(hidden_activation, str):
            hidden_activation = activation_from_string(hidden_activation)
        self.hidden_activation = hidden_activation

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.normalization_type = normalization_type

        self.layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.pool_type = pool_type

        for i, (out_channels, kernel_size, stride, padding) in enumerate(
                zip(n_channels, kernel_sizes, strides, paddings)
        ):
            deconv = nn.ConvTranspose2d(input_channels,
                                        out_channels,
                                        kernel_size,
                                        stride=stride,
                                        padding=padding)
            if hidden_init:
                hidden_init(deconv.weight)

            layer = deconv
            self.layers.append(layer)
            input_channels = out_channels

            if pool_type == 'max2d':
                if pool_sizes[i] > 1:
                    self.pool_layers.append(
                        nn.MaxUnpool2d(
                            kernel_size=pool_sizes[i],
                            stride=pool_strides[i],
                            padding=pool_paddings[i],
                        )
                    )
                else:
                    self.pool_layers.append(None)

        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )
        for layer in self.layers:
            test_mat = layer(test_mat)
            if self.normalization_type == 'batch':
                self.norm_layers.append(
                    nn.BatchNorm2d(test_mat.shape[1])
                )
            if self.normalization_type == 'layer':
                self.norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))
        self.output_shape = test_mat.shape[1:]  # ignore batch dim

    def forward(self, input):
        h = input.view(
            -1, self.input_channels, self.input_width,
            self.input_height
        )
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if self.normalization_type != 'none':
                h = self.norm_layers[i](h)
            if self.pool_type != 'none':
                if self.pool_layers[i]:
                    h = self.pool_layers[i](h)
            h = self.hidden_activation(h)
        return h
