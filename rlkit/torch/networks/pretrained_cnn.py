import numpy as np
import torch
import torchvision.models as models
from torch import nn as nn

from rlkit.pythonplusplus import identity
from rlkit.torch.core import PyTorchModule


class PretrainedCNN(PyTorchModule):
    # Uses a pretrained CNN architecture from torchvision
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            hidden_sizes=None,
            added_fc_input_size=0,
            batch_norm_fc=False,
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            output_conv_channels=False,
            model_architecture=models.resnet18,
            model_pretrained=True,
            model_freeze=False,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.batch_norm_fc = batch_norm_fc
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.output_conv_channels = output_conv_channels

        self.pretrained_model = nn.Sequential(*list(model_architecture(
            pretrained=model_pretrained).children())[:-1])
        if model_freeze:
            for child in self.pretrained_model.children():
                for param in child.parameters():
                    param.requires_grad = False
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        # use torch rather than ptu because initially the model is on CPU
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )
        # find output dim of conv_layers by trial and add norm conv layers
        test_mat = self.pretrained_model(test_mat)

        self.conv_output_flat_size = int(np.prod(test_mat.shape))
        if self.output_conv_channels:
            self.last_fc = None
        else:
            fc_input_size = self.conv_output_flat_size
            # used only for injecting input directly into fc layers
            fc_input_size += added_fc_input_size
            for idx, hidden_size in enumerate(hidden_sizes):
                fc_layer = nn.Linear(fc_input_size, hidden_size)
                fc_input_size = hidden_size

                fc_layer.weight.data.uniform_(-init_w, init_w)
                fc_layer.bias.data.uniform_(-init_w, init_w)

                self.fc_layers.append(fc_layer)

                if self.batch_norm_fc:
                    norm_layer = nn.BatchNorm1d(hidden_size)
                    self.fc_norm_layers.append(norm_layer)

            self.last_fc = nn.Linear(fc_input_size, output_size)
            self.last_fc.weight.data.uniform_(-init_w, init_w)
            self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_last_activations=False):
        conv_input = input.narrow(start=0,
                                  length=self.conv_input_length,
                                  dim=1).contiguous()
        # reshape from batch of flattened images into (channels, w, h)
        h = conv_input.view(conv_input.shape[0],
                            self.input_channels,
                            self.input_height,
                            self.input_width)

        h = self.apply_forward_conv(h)

        if self.output_conv_channels:
            return h

        # flatten channels for fc layers
        h = h.view(h.size(0), -1)
        if self.added_fc_input_size != 0:
            extra_fc_input = input.narrow(
                start=self.conv_input_length,
                length=self.added_fc_input_size,
                dim=1,
            )
            h = torch.cat((h, extra_fc_input), dim=1)
        h = self.apply_forward_fc(h)

        if return_last_activations:
            return h
        return self.output_activation(self.last_fc(h))

    def apply_forward_conv(self, h):
        return self.pretrained_model(h)

    def apply_forward_fc(self, h):
        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            if self.batch_norm_fc:
                h = self.fc_norm_layers[i](h)
            h = self.hidden_activation(h)
        return h


