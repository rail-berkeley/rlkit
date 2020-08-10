import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule


class FeatPointMlp(PyTorchModule):
    def __init__(
            self,
            downsample_size,
            input_channels,
            num_feat_points,
            temperature=1.0,
            init_w=1e-3,
            input_size=32,
            hidden_init=ptu.fanin_init,
            output_activation=identity,
    ):
        super().__init__()

        self.downsample_size = downsample_size
        self.temperature = temperature
        self.num_feat_points = num_feat_points
        self.hidden_init = hidden_init
        self.output_activation = output_activation
        self.input_channels = input_channels
        self.input_size = input_size

        #        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(input_channels, 48, kernel_size=5, stride=2)
        #        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(48, 48, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(48, self.num_feat_points, kernel_size=5, stride=1)

        test_mat = ptu.zeros(1, self.input_channels, self.input_size, self.input_size)
        test_mat = self.conv1(test_mat)
        test_mat = self.conv2(test_mat)
        test_mat = self.conv3(test_mat)
        self.out_size = int(np.prod(test_mat.shape))
        self.fc1 = nn.Linear(2 * self.num_feat_points, 400)
        self.fc2 = nn.Linear(400, 300)
        self.last_fc = nn.Linear(300, self.input_channels * self.downsample_size * self.downsample_size)

        self.init_weights(init_w)
        self.i = 0

    def init_weights(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, input):
        h = self.encoder(input)
        out = self.decoder(h)
        return out

    def encoder(self, input):
        x = input.contiguous().view(-1, self.input_channels, self.input_size, self.input_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        d = int((self.out_size // self.num_feat_points) ** (1 / 2))
        x = x.view(-1, self.num_feat_points, d * d)
        x = F.softmax(x / self.temperature, 2)
        x = x.view(-1, self.num_feat_points, d, d)

        maps_x = torch.sum(x, 2)
        maps_y = torch.sum(x, 3)

        weights = ptu.from_numpy(np.arange(d) / (d + 1))

        fp_x = torch.sum(maps_x * weights, 2)
        fp_y = torch.sum(maps_y * weights, 2)

        x = torch.cat([fp_x, fp_y], 1)
        #        h = x.view(-1, 2, self.num_feat_points).transpose(1, 2).contiguous().view(-1, self.num_feat_points * 2)
        h = x.view(-1, self.num_feat_points * 2)
        return h

    def decoder(self, input):
        h = input
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.last_fc(h)
        return h

    def history_encoder(self, input, history_length):
        input = input.contiguous().view(-1,
                                        self.input_channels,
                                        self.input_size,
                                        self.input_size)
        latent = self.encoder(input)

        assert latent.shape[0] % history_length == 0
        n_samples = latent.shape[0] // history_length
        latent = latent.view(n_samples, -1)
        return latent

