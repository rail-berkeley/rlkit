import math
import numbers
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import jit, nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.tensor import Tensor

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.model_based.dreamer.actor_models import OneHotDist
from rlkit.torch.model_based.dreamer.conv_networks import CNN, DCNN
from rlkit.torch.model_based.dreamer.mlp import Mlp


class WorldModel(jit.ScriptModule):
    def __init__(
        self,
        action_dim,
        image_shape,
        stochastic_state_size=60,
        deterministic_state_size=400,
        embedding_size=1024,
        rssm_hidden_size=400,
        model_hidden_size=400,
        model_act=F.elu,
        depth=32,
        conv_act=F.relu,
        reward_num_layers=2,
        pred_discount_num_layers=3,
        gru_layer_norm=False,
        discrete_latents=False,
        discrete_latent_size=32,
        std_act="softplus",
    ):
        super().__init__()
        self.image_shape = image_shape
        self.model_act = model_act
        self.stochastic_state_size = stochastic_state_size
        self.deterministic_state_size = deterministic_state_size
        self.discrete_latents = discrete_latents
        self.discrete_latent_size = discrete_latent_size
        self.depth = depth
        self.conv_act = conv_act
        if discrete_latents:
            img_and_obs_step_mlp_output_size = (
                stochastic_state_size * discrete_latent_size
            )
            full_stochastic_state_size = img_and_obs_step_mlp_output_size
        else:
            img_and_obs_step_mlp_output_size = 2 * stochastic_state_size
            full_stochastic_state_size = stochastic_state_size
        self.feature_size = deterministic_state_size + full_stochastic_state_size

        self.obs_step_mlp = Mlp(
            hidden_sizes=[rssm_hidden_size],
            input_size=embedding_size + deterministic_state_size,
            output_size=img_and_obs_step_mlp_output_size,
            hidden_activation=model_act,
            hidden_init=torch.nn.init.xavier_uniform_,
        )
        self.action_step_feature_extractor = torch.nn.Linear(
            full_stochastic_state_size + action_dim, deterministic_state_size
        )
        torch.nn.init.xavier_uniform_(self.action_step_feature_extractor.weight)
        self.action_step_feature_extractor.bias.data.fill_(0)
        self.action_step_mlp = Mlp(
            hidden_sizes=[rssm_hidden_size],
            input_size=deterministic_state_size,
            output_size=img_and_obs_step_mlp_output_size,
            hidden_activation=model_act,
            hidden_init=torch.nn.init.xavier_uniform_,
        )
        if gru_layer_norm:
            self.rnn = LayerNormGRUCell(
                deterministic_state_size, deterministic_state_size
            )
        else:
            self.rnn = torch.nn.GRUCell(
                deterministic_state_size, deterministic_state_size
            )
        self.conv_encoder = CNN(
            input_width=self.image_shape[2],
            input_height=self.image_shape[1],
            input_channels=self.image_shape[0],
            kernel_sizes=[4] * 4,
            n_channels=[depth, depth * 2, depth * 4, depth * 8],
            strides=[2] * 4,
            paddings=[0] * 4,
            hidden_activation=conv_act,
            hidden_init=torch.nn.init.xavier_uniform_,
        )

        self.conv_decoder = DCNN(
            fc_input_size=self.feature_size,
            hidden_sizes=[],
            deconv_input_width=1,
            deconv_input_height=1,
            deconv_input_channels=depth * 32,
            deconv_output_kernel_size=6,
            deconv_output_strides=2,
            deconv_output_channels=self.image_shape[0],
            kernel_sizes=[5, 5, 6],
            n_channels=[depth * 4, depth * 2, depth * 1],
            strides=[2] * 3,
            paddings=[0] * 3,
            hidden_activation=conv_act,
            hidden_init=torch.nn.init.xavier_uniform_,
        )

        self.reward = Mlp(
            hidden_sizes=[model_hidden_size] * reward_num_layers,
            input_size=self.feature_size,
            output_size=1,
            hidden_activation=model_act,
            hidden_init=torch.nn.init.xavier_uniform_,
        )
        self.pred_discount = Mlp(
            hidden_sizes=[model_hidden_size] * pred_discount_num_layers,
            input_size=self.feature_size,
            output_size=1,
            hidden_activation=model_act,
            hidden_init=torch.nn.init.xavier_uniform_,
        )
        self.std_act = std_act

    @jit.script_method
    def compute_std(self, std):
        if self.std_act == "softplus":
            std = F.softplus(std)
        elif self.std_act == "sigmoid2":
            std = 2 * torch.sigmoid(std / 2)
        return std + 0.1

    @jit.script_method
    def obs_step(
        self, prev_state: Dict[str, Tensor], prev_action: Tensor, embed: Tensor
    ):
        prior = self.action_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        x = self.obs_step_mlp(x)
        if self.discrete_latents:
            logits = x.reshape(
                list(x.shape[:-1])
                + [self.stochastic_state_size, self.discrete_latent_size]
            )
            stoch = self.get_discrete_stochastic_state(logits)
            post = {"logits": logits, "stoch": stoch, "deter": prior["deter"]}
        else:
            mean, std = x.split(self.stochastic_state_size, -1)
            std = self.compute_std(std)
            stoch = (
                torch.randn(mean.shape, device=ptu.device, dtype=mean.dtype) * std
                + mean
            )
            post = {"mean": mean, "std": std, "stoch": stoch, "deter": prior["deter"]}
        return post, prior

    @jit.ignore
    def get_discrete_stochastic_state(self, logits):
        stoch = self.get_dist(logits, logits, latent=True).rsample()
        return stoch

    @jit.script_method
    def action_step(self, prev_state: Dict[str, Tensor], prev_action: Tensor):
        prev_stoch = prev_state["stoch"]
        if self.discrete_latents:
            shape = list(prev_stoch.shape[:-2]) + [
                self.stochastic_state_size * self.discrete_latent_size
            ]
            prev_stoch = prev_stoch.reshape(shape)

        x = torch.cat([prev_stoch, prev_action], -1)
        x = self.model_act(self.action_step_feature_extractor(x))
        deter_new = self.rnn(x, prev_state["deter"])
        x = self.action_step_mlp(deter_new)
        if self.discrete_latents:
            logits = x.reshape(
                list(x.shape[:-1])
                + [self.stochastic_state_size, self.discrete_latent_size]
            )
            stoch = self.get_discrete_stochastic_state(logits)
            prior = {"logits": logits, "stoch": stoch, "deter": deter_new}
        else:
            mean, std = x.split(self.stochastic_state_size, -1)
            std = F.softplus(std) + 0.1
            stoch = (
                torch.randn(mean.shape, device=ptu.device, dtype=mean.dtype) * std
                + mean
            )
            prior = {"mean": mean, "std": std, "stoch": stoch, "deter": deter_new}
        return prior

    @jit.script_method
    def forward_batch(
        self,
        path_length: int,
        action: Tensor,
        embed: Tensor,
        post: Dict[str, List[Tensor]],
        prior: Dict[str, List[Tensor]],
        state: Dict[str, Tensor],
    ):
        for i in range(path_length):
            (post_params, prior_params,) = self.obs_step(
                state,
                action[:, i],
                embed[:, i],
            )
            for k in post.keys():
                post[k].append(post_params[k].unsqueeze(1))

            for k in prior.keys():
                prior[k].append(prior_params[k].unsqueeze(1))
            state = post_params
        return post, prior

    def forward(self, obs, action):
        original_batch_size = obs.shape[0]
        state = self.initial(original_batch_size)
        path_length = obs.shape[1]
        if self.discrete_latents:
            post, prior = (
                dict(logits=[], stoch=[], deter=[]),
                dict(logits=[], stoch=[], deter=[]),
            )
        else:
            post, prior = (
                dict(mean=[], std=[], stoch=[], deter=[]),
                dict(mean=[], std=[], stoch=[], deter=[]),
            )
        obs = obs.transpose(1, 0).reshape(-1, obs.shape[-1])
        embed = self.encode(obs)
        embedding_size = embed.shape[1]
        embed = embed.reshape(
            path_length, original_batch_size, embedding_size
        ).transpose(1, 0)
        post, prior = self.forward_batch(path_length, action, embed, post, prior, state)

        for k in post.keys():
            post[k] = torch.cat(post[k], dim=1)

        for k in prior.keys():
            prior[k] = torch.cat(prior[k], dim=1)

        feat = self.get_features(post)
        feat = feat.transpose(1, 0).reshape(-1, feat.shape[-1])
        images = self.decode(feat)
        rewards = self.reward(feat)
        pred_discounts = self.pred_discount(feat)

        if self.discrete_latents:
            post_dist = self.get_dist(post["logits"], None, latent=True)
            prior_dist = self.get_dist(prior["logits"], None, latent=True)
        else:
            post_dist = self.get_dist(post["mean"], post["std"], latent=True)
            prior_dist = self.get_dist(prior["mean"], prior["std"], latent=True)
        image_dist = self.get_dist(images, ptu.ones_like(images), dims=3)
        reward_dist = self.get_dist(rewards, ptu.ones_like(rewards))
        pred_discount_dist = self.get_dist(pred_discounts, None, normal=False)
        return (
            post,
            prior,
            post_dist,
            prior_dist,
            image_dist,
            reward_dist,
            pred_discount_dist,
            embed,
        )

    def get_features(self, state: Dict[str, Tensor]):
        stoch = state["stoch"]
        if self.discrete_latents:
            shape = list(stoch.shape[:-2]) + [
                self.stochastic_state_size * self.discrete_latent_size
            ]
            stoch = stoch.reshape(shape)
        return torch.cat([state["deter"], stoch], -1)

    def get_dist(self, mean, std, dims=1, normal=True, latent=False):
        if normal:
            if latent and self.discrete_latents:
                dist = Independent(OneHotDist(logits=mean), dims)
                return dist
            return Independent(Normal(mean, std), dims)
        else:
            return Independent(Bernoulli(logits=mean), dims)

    def get_detached_dist(self, params, dims=1, normal=True, latent=False):
        if self.discrete_latents:
            mean = params["logits"]
            std = params["logits"]
        else:
            mean = params["mean"]
            std = params["std"]
        return self.get_dist(mean.detach(), std.detach(), dims, normal, latent)

    @jit.script_method
    def encode(self, obs):
        return self.conv_encoder(self.preprocess(obs))

    @jit.script_method
    def decode(self, feat):
        return self.conv_decoder(feat)

    @jit.script_method
    def preprocess(self, obs):
        obs = obs / 255.0 - 0.5
        return obs

    def initial(self, batch_size):
        if self.discrete_latents:
            state = dict(
                logits=ptu.zeros(
                    [batch_size, self.stochastic_state_size, self.discrete_latent_size],
                ),
                stoch=ptu.zeros(
                    [batch_size, self.stochastic_state_size, self.discrete_latent_size]
                ),
                deter=ptu.zeros([batch_size, self.deterministic_state_size]),
            )
        else:
            state = dict(
                mean=ptu.zeros([batch_size, self.stochastic_state_size]),
                std=ptu.zeros([batch_size, self.stochastic_state_size]),
                stoch=ptu.zeros([batch_size, self.stochastic_state_size]),
                deter=ptu.zeros([batch_size, self.deterministic_state_size]),
            )
        return state


class LayerNorm(jit.ScriptModule):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    @jit.script_method
    def compute_layernorm_stats(self, input):
        mu = input.mean(-1, keepdim=True)
        sigma = input.std(-1, keepdim=True, unbiased=False)
        return mu, sigma

    @jit.script_method
    def forward(self, input):
        mu, sigma = self.compute_layernorm_stats(input)
        return (input - mu) / sigma * self.weight + self.bias


class LayerNormGRUCell(jit.ScriptModule):
    def __init__(self, input_size, output_size):
        super(LayerNormGRUCell, self).__init__()
        hidden_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        bias = True
        num_chunks = 3
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(num_chunks * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(num_chunks * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()

        self._size = output_size
        self._act = torch.tanh
        self._update_bias = -1
        self._layer = nn.Linear(input_size * 2, 3 * output_size, bias=True)
        self._norm = nn.LayerNorm((output_size * 3))

    @jit.script_method
    def forward(self, inputs, hx):
        parts = self._layer(torch.cat([inputs, hx], -1))
        parts = self._norm(parts)
        reset, cand, update = torch.split(parts, self._size, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * hx
        return output

    def extra_repr(self) -> str:
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        if "nonlinearity" in self.__dict__ and self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        return s.format(**self.__dict__)

    def check_forward_input(self, input: Tensor) -> None:
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size
                )
            )

    def check_forward_hidden(
        self, input: Tensor, hx: Tensor, hidden_label: str = ""
    ) -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)
                )
            )

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size
                )
            )

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)
