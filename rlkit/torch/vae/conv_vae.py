import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
import numpy as np
from rlkit.torch.conv_networks import CNN, DCNN
from rlkit.torch.vae.vae_base import GaussianLatentVAE

###### DEFAULT ARCHITECTURES #########

imsize48_default_architecture = dict(
    conv_args=dict(
        kernel_sizes=[5, 3, 3],
        n_channels=[16, 32, 64],
        strides=[3, 2, 2],
    ),
    conv_kwargs=dict(
        hidden_sizes=[],
        batch_norm_conv=False,
        batch_norm_fc=False,
    ),
    deconv_args=dict(
        hidden_sizes=[],

        deconv_input_width=3,
        deconv_input_height=3,
        deconv_input_channels=64,

        deconv_output_kernel_size=6,
        deconv_output_strides=3,
        deconv_output_channels=3,

        kernel_sizes=[3, 3],
        n_channels=[32, 16],
        strides=[2, 2],
    ),
    deconv_kwargs=dict(
        batch_norm_deconv=False,
        batch_norm_fc=False,
    )
)

imsize48_default_architecture_with_more_hidden_layers = dict(
    conv_args=dict(
        kernel_sizes=[5, 3, 3],
        n_channels=[16, 32, 64],
        strides=[3, 2, 2],
    ),
    conv_kwargs=dict(
        hidden_sizes=[500, 300, 150],
    ),
    deconv_args=dict(
        hidden_sizes=[150, 300, 500],

        deconv_input_width=3,
        deconv_input_height=3,
        deconv_input_channels=64,

        deconv_output_kernel_size=6,
        deconv_output_strides=3,
        deconv_output_channels=3,

        kernel_sizes=[3, 3],
        n_channels=[32, 16],
        strides=[2, 2],
    ),
    deconv_kwargs=dict(
    )
)

imsize84_default_architecture = dict(
    conv_args=dict(
        kernel_sizes=[5, 5, 5],
        n_channels=[16, 32, 32],
        strides=[3, 3, 3],
    ),
    conv_kwargs=dict(
        hidden_sizes=[],
        batch_norm_conv=True,
        batch_norm_fc=False,
    ),
    deconv_args=dict(
        hidden_sizes=[],

        deconv_input_width=2,
        deconv_input_height=2,
        deconv_input_channels=32,

        deconv_output_kernel_size=6,
        deconv_output_strides=3,
        deconv_output_channels=3,

        kernel_sizes=[5, 6],
        n_channels=[32, 16],
        strides=[3, 3],
    ),
    deconv_kwargs=dict(
        batch_norm_deconv=False,
        batch_norm_fc=False,
    )
)


class ConvVAE(GaussianLatentVAE):
    def __init__(
            self,
            representation_size,
            architecture,

            encoder_class=CNN,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',

            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            hidden_init=ptu.fanin_init,
    ):
        """

        :param representation_size:
        :param conv_args:
        must be a dictionary specifying the following:
            kernel_sizes
            n_channels
            strides
        :param conv_kwargs:
        a dictionary specifying the following:
            hidden_sizes
            batch_norm
        :param deconv_args:
        must be a dictionary specifying the following:
            hidden_sizes
            deconv_input_width
            deconv_input_height
            deconv_input_channels
            deconv_output_kernel_size
            deconv_output_strides
            deconv_output_channels
            kernel_sizes
            n_channels
            strides
        :param deconv_kwargs:
            batch_norm
        :param encoder_class:
        :param decoder_class:
        :param decoder_output_activation:
        :param decoder_distribution:
        :param input_channels:
        :param imsize:
        :param init_w:
        :param min_variance:
        :param hidden_init:
        """
        super().__init__(representation_size)
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize * self.imsize * self.input_channels

        conv_args, conv_kwargs, deconv_args, deconv_kwargs = \
            architecture['conv_args'], architecture['conv_kwargs'], \
            architecture['deconv_args'], architecture['deconv_kwargs']
        conv_output_size = deconv_args['deconv_input_width'] * \
                           deconv_args['deconv_input_height'] * \
                           deconv_args['deconv_input_channels']

        self.encoder = encoder_class(
            **conv_args,
            paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
            input_height=self.imsize,
            input_width=self.imsize,
            input_channels=self.input_channels,
            output_size=conv_output_size,
            init_w=init_w,
            hidden_init=hidden_init,
            **conv_kwargs)

        self.fc1 = nn.Linear(self.encoder.output_size, representation_size)
        self.fc2 = nn.Linear(self.encoder.output_size, representation_size)

        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)

        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

        self.decoder = decoder_class(
            **deconv_args,
            fc_input_size=representation_size,
            init_w=init_w,
            output_activation=decoder_output_activation,
            paddings=np.zeros(len(deconv_args['kernel_sizes']), dtype=np.int64),
            hidden_init=hidden_init,
            **deconv_kwargs)

        self.epoch = 0
        self.decoder_distribution = decoder_distribution

    def encode(self, input):
        h = self.encoder(input)
        mu = self.fc1(h)
        if self.log_min_variance is None:
            logvar = self.fc2(h)
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))
        return (mu, logvar)

    def decode(self, latents):
        decoded = self.decoder(latents).view(-1,
                                             self.imsize * self.imsize * self.input_channels)
        if self.decoder_distribution == 'bernoulli':
            return decoded, [decoded]
        elif self.decoder_distribution == 'gaussian_identity_variance':
            return torch.clamp(decoded, 0, 1), [torch.clamp(decoded, 0, 1),
                                                torch.ones_like(decoded)]
        else:
            raise NotImplementedError('Distribution {} not supported'.format(
                self.decoder_distribution))

    def logprob(self, inputs, obs_distribution_params):
        if self.decoder_distribution == 'bernoulli':
            inputs = inputs.narrow(start=0, length=self.imlength,
                                   dim=1).contiguous().view(-1, self.imlength)
            log_prob = - F.binary_cross_entropy(
                obs_distribution_params[0],
                inputs,
                reduction='elementwise_mean'
            ) * self.imlength
            return log_prob
        if self.decoder_distribution == 'gaussian_identity_variance':
            inputs = inputs.narrow(start=0, length=self.imlength,
                                   dim=1).contiguous().view(-1, self.imlength)
            log_prob = -1 * F.mse_loss(inputs, obs_distribution_params[0],
                                       reduction='elementwise_mean')
            return log_prob
        else:
            raise NotImplementedError('Distribution {} not supported'.format(
                self.decoder_distribution))
