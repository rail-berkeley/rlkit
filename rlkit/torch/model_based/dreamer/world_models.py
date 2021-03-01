import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.model_based.dreamer.actor_models import OneHotDist
from rlkit.torch.model_based.dreamer.conv_networks import CNN, DCNN
from rlkit.torch.model_based.dreamer.mlp import Mlp


class WorldModel(PyTorchModule):
    def __init__(
        self,
        action_dim,
        image_shape,
        env,
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
        use_depth_wise_separable_conv=False,
        gru_layer_norm=False,
        discrete_latents=False,
        discrete_latent_size=32,
        use_per_primitive_feature_extractor=False,
        std_act="softplus",
    ):
        super().__init__()
        self.image_shape = image_shape
        self.model_act = model_act
        self.stochastic_state_size = stochastic_state_size
        self.deterministic_state_size = deterministic_state_size
        self.discrete_latents = discrete_latents
        self.discrete_latent_size = discrete_latent_size
        self.env = env
        self.depth = depth
        self.use_depth_wise_separable_conv = use_depth_wise_separable_conv
        self.conv_act = conv_act
        if discrete_latents:
            img_and_obs_step_mlp_output_size = (
                stochastic_state_size * discrete_latent_size
            )
            full_stochastic_state_size = img_and_obs_step_mlp_output_size
            feature_size = full_stochastic_state_size + deterministic_state_size
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
        self.use_per_primitive_feature_extractor = use_per_primitive_feature_extractor
        if self.use_per_primitive_feature_extractor:
            self.action_step_feature_extractor = torch.nn.ModuleDict()
            for (
                k,
                v,
            ) in env.primitive_name_to_action_idx.items():
                if type(v) == int:
                    len_v = 1
                else:
                    len_v = len(v)
                layer = torch.nn.Linear(
                    full_stochastic_state_size + len_v, deterministic_state_size
                )
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
                self.action_step_feature_extractor[k] = layer
        else:
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
            self.rnn = GRUCell(
                deterministic_state_size, deterministic_state_size, norm=True
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
            use_depth_wise_separable_conv=use_depth_wise_separable_conv,
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
            use_depth_wise_separable_conv=use_depth_wise_separable_conv,
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

    def compute_std(self, std):
        if self.std_act == "softplus":
            std = F.softplus(std)
        elif self.std_act == "sigmoid2":
            std = 2 * torch.sigmoid(std / 2)
        return std + 0.1

    def obs_step(self, prev_state, prev_action, embed):
        prior = self.action_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        x = self.obs_step_mlp(x)
        if self.discrete_latents:
            logits = x.reshape(
                list(x.shape[:-1])
                + [self.stochastic_state_size, self.discrete_latent_size]
            )
            stoch = self.get_dist(logits, logits, latent=True).rsample()
            post = {"logits": logits, "stoch": stoch, "deter": prior["deter"]}
        else:
            mean, std = x.split(self.stochastic_state_size, -1)
            std = self.compute_std(std)
            stoch = self.get_dist(mean, std, latent=True).rsample()
            post = {"mean": mean, "std": std, "stoch": stoch, "deter": prior["deter"]}
        return post, prior

    def action_step(self, prev_state, prev_action):
        prev_stoch = prev_state["stoch"]
        if self.discrete_latents:
            shape = list(prev_stoch.shape[:-2]) + [
                self.stochastic_state_size * self.discrete_latent_size
            ]
            prev_stoch = prev_stoch.reshape(shape)

        if self.use_per_primitive_feature_extractor:
            primitive_indices, primitive_args = (
                torch.argmax(prev_action[:, : self.env.num_primitives], dim=1),
                prev_action[:, self.env.num_primitives :],
            )
            output = {}
            for k, v in self.env.primitive_name_to_action_idx.items():
                if type(v) is int:
                    v = [v]
                primitive_action = primitive_args[:, v]
                x = torch.cat([prev_stoch, primitive_action], -1)
                output[k] = self.model_act(self.action_step_feature_extractor[k](x))
            primitive_names = [
                self.env.primitive_idx_to_name[primitive_idx]
                for primitive_idx in primitive_indices.cpu().detach().numpy().tolist()
            ]
            x = []
            for i, primitive_name in enumerate(primitive_names):
                x.append(output[primitive_name][i : i + 1])
            x = torch.cat(x, dim=0)
        else:
            x = torch.cat([prev_stoch, prev_action], -1)
            x = self.model_act(self.action_step_feature_extractor(x))
        deter_new = self.rnn(x, prev_state["deter"])
        x = self.action_step_mlp(deter_new)
        if self.discrete_latents:
            logits = x.reshape(
                list(x.shape[:-1])
                + [self.stochastic_state_size, self.discrete_latent_size]
            )
            stoch = self.get_dist(logits, None, latent=True).rsample()
            prior = {"logits": logits, "stoch": stoch, "deter": deter_new}
        else:
            mean, std = x.split(self.stochastic_state_size, -1)
            std = F.softplus(std) + 0.1
            stoch = self.get_dist(mean, std, latent=True).rsample()
            prior = {"mean": mean, "std": std, "stoch": stoch, "deter": deter_new}
        return prior

    def forward_batch(
        self,
        embed,
        action,
        state=None,
    ):
        post_params, prior_params = self.obs_step(state, action, embed)
        feat = self.get_feat(post_params)
        image_params = self.decode(feat)
        reward_params = self.reward(feat)
        pred_discount_params = self.pred_discount(feat)
        return (
            post_params,
            prior_params,
            image_params,
            reward_params,
            pred_discount_params,
        )

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
        images, rewards, pred_discounts = [], [], []
        obs = torch.cat([obs[:, i, :] for i in range(obs.shape[1])])
        embed = self.encode(obs)
        embedding_size = embed.shape[1]
        embed = torch.cat(
            [
                embed[
                    i * original_batch_size : (i + 1) * original_batch_size, :
                ].reshape(original_batch_size, 1, embedding_size)
                for i in range(path_length)
            ],
            dim=1,
        )
        for i in range(path_length):
            (
                post_params,
                prior_params,
                image_params,
                reward_params,
                pred_discount_params,
            ) = self.forward_batch(embed[:, i], action[:, i], state)
            images.append(image_params)
            rewards.append(reward_params)
            pred_discounts.append(pred_discount_params)
            for k in post.keys():
                post[k].append(post_params[k].unsqueeze(1))

            for k in prior.keys():
                prior[k].append(prior_params[k].unsqueeze(1))
            state = post_params

        images = torch.cat(images)
        rewards = torch.cat(rewards)
        pred_discounts = torch.cat(pred_discounts)
        for k in post.keys():
            post[k] = torch.cat(post[k], dim=1)

        for k in prior.keys():
            prior[k] = torch.cat(prior[k], dim=1)

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

    def get_feat(self, state):
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

    def get_detached_dist(self, mean, std, dims=1, normal=True, latent=False):
        return self.get_dist(mean.detach(), std.detach(), dims, normal, latent)

    def encode(self, obs):
        return self.conv_encoder(self.preprocess(obs))

    def decode(self, feat):
        return self.conv_decoder(feat)

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


class StateConcatObsWorldModel(WorldModel):
    def __init__(self, *args, state_output_size=1, **kwargs):
        super(StateConcatObsWorldModel, self).__init__(*args, **kwargs)
        self.conv_decoder = DCNN(
            fc_input_size=self.feature_size,
            hidden_sizes=[],
            deconv_input_width=1,
            deconv_input_height=1,
            deconv_input_channels=self.depth * 32,
            deconv_output_kernel_size=6,
            deconv_output_strides=2,
            deconv_output_channels=self.image_shape[0],
            kernel_sizes=[5, 5, 6],
            n_channels=[self.depth * 4, self.depth * 2, self.depth * 1],
            strides=[2] * 3,
            paddings=[0] * 3,
            hidden_activation=self.conv_act,
            hidden_init=torch.nn.init.xavier_uniform_,
            use_depth_wise_separable_conv=self.use_depth_wise_separable_conv,
            include_vector_output=True,
            vector_output_size=state_output_size,
        )

    def preprocess(self, obs):
        image_obs, _ = (
            obs[:, : np.prod(self.image_shape)],
            obs[:, np.prod(self.image_shape) :],
        )
        image_obs = super(StateConcatObsWorldModel, self).preprocess(image_obs)
        return image_obs

    def encode(self, obs):
        image_obs, state = (
            obs[:, : np.prod(self.image_shape)],
            obs[:, np.prod(self.image_shape) :],
        )
        image_obs = self.preprocess(obs)
        encoded_obs = self.conv_encoder(image_obs)
        # print(
        #     "Mean Encoded {}, Mean state {}".format(
        #         encoded_obs.mean().item(), state.mean().item()
        #     )
        # )
        # print(
        #     "Std Encoded {}, Std state {}".format(
        #         encoded_obs.std().item(), state.std().item()
        #     )
        # )
        # print(
        #     "Max Encoded {}, Max state {}".format(
        #         encoded_obs.max().item(), state.max().item()
        #     )
        # )
        # print(
        #     "Min Encoded {}, Min state {}".format(
        #         encoded_obs.min().item(), state.min().item()
        #     )
        # )
        # print()

        latent = torch.cat((encoded_obs, state), dim=1)
        return latent

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
        images, rewards, pred_discounts, states = [], [], [], []
        obs = torch.cat([obs[:, i, :] for i in range(obs.shape[1])])
        embed = self.encode(obs)
        embedding_size = embed.shape[1]
        embed = torch.cat(
            [
                embed[
                    i * original_batch_size : (i + 1) * original_batch_size, :
                ].reshape(original_batch_size, 1, embedding_size)
                for i in range(path_length)
            ],
            dim=1,
        )
        for i in range(path_length):
            (
                post_params,
                prior_params,
                image_params,
                reward_params,
                pred_discount_params,
            ) = self.forward_batch(embed[:, i], action[:, i], state)
            images.append(image_params[0])
            states.append(image_params[1])
            rewards.append(reward_params)
            pred_discounts.append(pred_discount_params)
            for k in post.keys():
                post[k].append(post_params[k].unsqueeze(1))

            for k in prior.keys():
                prior[k].append(prior_params[k].unsqueeze(1))
            state = post_params

        images = torch.cat(images)
        rewards = torch.cat(rewards)
        states = torch.cat(states)
        pred_discounts = torch.cat(pred_discounts)
        for k in post.keys():
            post[k] = torch.cat(post[k], dim=1)

        for k in prior.keys():
            prior[k] = torch.cat(prior[k], dim=1)

        if self.discrete_latents:
            post_dist = self.get_dist(post["logits"], None, latent=True)
            prior_dist = self.get_dist(prior["logits"], None, latent=True)
        else:
            post_dist = self.get_dist(post["mean"], post["std"], latent=True)
            prior_dist = self.get_dist(prior["mean"], prior["std"], latent=True)
        image_dist = self.get_dist(images, ptu.ones_like(images), dims=3)
        reward_dist = self.get_dist(rewards, ptu.ones_like(rewards))
        pred_discount_dist = self.get_dist(pred_discounts, None, normal=False)
        state_dist = self.get_dist(states, ptu.ones_like(states))
        return (
            post,
            prior,
            post_dist,
            prior_dist,
            (image_dist, state_dist),
            reward_dist,
            pred_discount_dist,
            embed,
        )


class GRUCell(nn.GRUCell):
    def __init__(
        self, input_size, output_size, norm=False, act=torch.tanh, update_bias=-1
    ):
        super(GRUCell, self).__init__(input_size, output_size)
        self._size = output_size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(input_size * 2, 3 * output_size, bias=norm is not None)
        if norm:
            self._norm = nn.LayerNorm((output_size * 3))

    def forward(self, inputs, hx=None):
        if hx is None:
            hx = torch.zeros(
                inputs.size(0),
                self.hidden_size,
                dtype=inputs.dtype,
                device=inputs.device,
            )
        parts = self._layer(torch.cat([inputs, hx], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, self._size, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * hx
        return output
