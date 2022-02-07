import math
import numbers
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, jit, nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from torch.nn import init
from torch.nn.parameter import Parameter

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.model_based.dreamer.conv_networks import CNN, DCNN
from rlkit.torch.model_based.dreamer.mlp import Mlp
from rlkit.torch.model_based.dreamer.utils import get_indexed_arr_from_batch_indices


class WorldModel(jit.ScriptModule):
    def __init__(
        self,
        action_dim,
        image_shape,
        stochastic_state_size=32,
        deterministic_state_size=200,
        rssm_hidden_size=200,
        model_hidden_size=400,
        model_act=nn.ELU,
        depth=48,
        conv_act=nn.ReLU,
        reward_num_layers=2,
        pred_discount_num_layers=3,
        gru_layer_norm=True,
        std_act="sigmoid2",
        use_prior_instead_of_posterior=False,
        reward_classifier=False,
    ):
        super().__init__()
        self.reward_classifier = reward_classifier
        self.use_prior_instead_of_posterior = use_prior_instead_of_posterior
        self.image_shape = image_shape
        self.stochastic_state_size = stochastic_state_size
        self.deterministic_state_size = deterministic_state_size
        self.depth = depth
        self.conv_act = conv_act
        img_and_obs_step_mlp_output_size = 2 * stochastic_state_size
        full_stochastic_state_size = stochastic_state_size
        self.feature_size = deterministic_state_size + full_stochastic_state_size
        embedding_size = depth * 32

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
        self.model_act = model_act(inplace=True)

    @jit.script_method
    def transform_std(self, std):
        """
        Transform the world model standard deviation using an activation.
        """
        if self.std_act == "softplus":
            std = F.softplus(std)
        elif self.std_act == "sigmoid2":
            std = 2 * torch.sigmoid(std / 2)
        return std + 0.1

    @jit.script_method
    def obs_step(
        self, prev_state: Dict[str, Tensor], prev_action: Tensor, embed: Tensor
    ):
        """
        Forward model dynamics (prior) then update using observation (posterior).

        :param prev_state: Dict
            mean: (batch_size, stoch_size)
            std: (batch_size, stoch_size)
            stoch: (batch_size, stoch_size)
            deter: (batch_size, deter_size)
        :param prev_action: (batch_size, action_dim)
        :param embed: (batch_size, embedding_size)

        :return: List[Dict]
            mean: (batch_size, stoch_size)
            std: (batch_size, stoch_size)
            stoch: (batch_size, stoch_size)
            deter: (batch_size, deter_size)
        """
        prior = self.action_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        x = self.obs_step_mlp(x)
        mean, std = x.split(self.stochastic_state_size, -1)
        std = self.transform_std(std)
        stoch = (
            torch.randn(mean.shape, device=ptu.device, dtype=mean.dtype) * std + mean
        )
        post = {"mean": mean, "std": std, "stoch": stoch, "deter": prior["deter"]}
        return post, prior

    @jit.script_method
    def action_step(self, prev_state: Dict[str, Tensor], prev_action: Tensor):
        """
        Forward model dynamics (compute prior).

        :param prev_state: Dict
            mean: (batch_size, stoch_size)
            std: (batch_size, stoch_size)
            stoch: (batch_size, stoch_size)
            deter: (batch_size, deter_size)
        :param prev_action: (batch_size, action_dim)

        :return: Dict
            mean: (batch_size, stoch_size)
            std: (batch_size, stoch_size)
            stoch: (batch_size, stoch_size)
            deter: (batch_size, deter_size)
        """
        prev_stoch = prev_state["stoch"]
        x = torch.cat([prev_stoch, prev_action], -1)
        x = self.model_act(self.action_step_feature_extractor(x))
        deter_new = self.rnn(x, prev_state["deter"])
        x = self.action_step_mlp(deter_new)
        mean, std = x.split(self.stochastic_state_size, -1)
        std = self.transform_std(std)
        stoch = (
            torch.randn(mean.shape, device=ptu.device, dtype=mean.dtype) * std + mean
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
        """
        Forward world model on batch of data using observation steps.

        :param path_length: int
        :param action: (batch_size, path_length, action_dim)
        :param embed: (batch_size, path_length, embedding_size)
        :param post: Dict
            mean: (batch_size, path_length, stoch_size)
            std: (batch_size, path_length, stoch_size)
            stoch: (batch_size, path_length, stoch_size)
            deter: (batch_size, path_length, deter_size)
        :param prior: Dict
            mean: (batch_size, path_length, stoch_size)
            std: (batch_size, path_length, stoch_size)
            stoch: (batch_size, path_length, stoch_size)
            deter: (batch_size, path_length, deter_size)
        :param state: Dict
            mean: (batch_size, stoch_size)
            std: (batch_size, stoch_size)
            stoch: (batch_size, stoch_size)
            deter: (batch_size, deter_size)

        :return: List[Dict]
            mean: (batch_size, path_length, stoch_size)
            std: (batch_size, path_length, stoch_size)
            stoch: (batch_size, path_length, stoch_size)
            deter: (batch_size, path_length, deter_size)
        """
        for step in range(path_length):
            (post_params, prior_params,) = self.obs_step(
                state,
                action[:, step],
                embed[:, step],
            )
            for key in post.keys():
                post[key].append(post_params[key].unsqueeze(1))

            for key in prior.keys():
                prior[key].append(prior_params[key].unsqueeze(1))
            state = post_params
        return post, prior

    def forward(self, obs, action):
        """
        Forward world model on trajectory.

        :param obs: (batch_size, path_length, obs_dim)
        :param action: (batch_size, path_length, action_dim)

        :return post: Dict
            mean: (batch_size, path_length, stoch_size)
            std: (batch_size, path_length, stoch_size)
            stoch: (batch_size, path_length, stoch_size)
            deter: (batch_size, path_length, deter_size)
        :return prior:
            mean: (batch_size, path_length, stoch_size)
            std: (batch_size, path_length, stoch_size)
            stoch: (batch_size, path_length, stoch_size)
            deter: (batch_size, path_length, deter_size)
        :return post_dist:
            mean: (batch_size*(path_length), stoch_size)
            std: (batch_size*(path_length), stoch_size)
        :return prior_dist:
            mean: (batch_size*(path_length), stoch_size)
            std: (batch_size*(path_length), stoch_size)
        :return image_dist:
            mean: (batch_size*(path_length), obs_dim)
        :return reward_dist:
            mean: (batch_size*(raps_path_length), 1)
        :return pred_discount_dist
            logits: (batch_size*(raps_path_length), 1)
        :return embed: (batch_size, path_length, embed_dim)
        """
        original_batch_size = obs.shape[0]
        state = self.initial(original_batch_size)
        path_length = obs.shape[1]
        post, prior = (
            dict(mean=[], std=[], stoch=[], deter=[]),
            dict(mean=[], std=[], stoch=[], deter=[]),
        )
        obs = obs.reshape(-1, obs.shape[-1])
        embed = self.encode(obs)
        embedding_size = embed.shape[1]
        embed = embed.reshape(original_batch_size, path_length, embedding_size)
        post, prior = self.forward_batch(
            path_length,
            action,
            embed,
            post,
            prior,
            state,
        )

        for key in post.keys():
            post[key] = torch.cat(post[key], dim=1)

        for key in prior.keys():
            prior[key] = torch.cat(prior[key], dim=1)

        if self.use_prior_instead_of_posterior:
            # In this case, o_hat_t depends on a_t-1 and o_t-1, reset obs decoded from null state + action.
            # This only works when first state is reset obs and never changes.
            feat = self.get_features(prior)
        else:
            feat = self.get_features(post)
        feat = feat.reshape(-1, feat.shape[-1])
        images = self.decode(feat)
        rewards = self.reward(feat)
        pred_discounts = self.pred_discount(feat)
        post_dist = self.get_dist(
            post["mean"].reshape(-1, post["mean"].shape[-1]),
            post["std"].reshape(-1, post["std"].shape[-1]),
        )
        prior_dist = self.get_dist(
            prior["mean"].reshape(-1, prior["mean"].shape[-1]),
            prior["std"].reshape(-1, prior["std"].shape[-1]),
        )
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
        return torch.cat([state["deter"], stoch], -1)

    def get_dist(self, mean, std, dims=1, normal=True):
        if normal:
            return Independent(Normal(mean, std), dims)
        else:
            return Independent(Bernoulli(logits=mean), dims)

    def get_detached_dist(self, params, dims=1, normal=True):
        mean = params["mean"]
        std = params["std"]
        return self.get_dist(mean.detach(), std.detach(), dims, normal)

    @jit.script_method
    def encode(self, obs):
        assert (
            len(obs.shape) == 2
        ), f"Input to encoder should only have two dimensions. Got {obs.shape}"
        return self.conv_encoder(self.preprocess(obs))

    @jit.script_method
    def decode(self, feat):
        assert (
            len(feat.shape) == 2
        ), f"Input to decoder should only have two dimensions. Got {feat.shape}"
        return self.conv_decoder(feat)

    @jit.script_method
    def preprocess(self, obs):
        obs = obs / 255.0 - 0.5
        return obs

    def initial(self, batch_size):
        """
        :param batch_size: int

        :return state: Dict
            mean: (batch_size, stoch_size)
            std: (batch_size, stoch_size)
            deter: (batch_size, deter_size)
            stoch: (batch_size, stoch_size)
        """
        state = dict(
            mean=ptu.zeros([batch_size, self.stochastic_state_size]),
            std=ptu.zeros([batch_size, self.stochastic_state_size]),
            stoch=ptu.zeros([batch_size, self.stochastic_state_size]),
            deter=ptu.zeros([batch_size, self.deterministic_state_size]),
        )
        return state


class LowlevelRAPSWorldModel(WorldModel):
    def __init__(self, *args, primitive_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.primitive_model = primitive_model

    @jit.script_method
    def forward_batch(
        self,
        path_length: int,
        action: Tuple[Tensor, Tensor],
        embed: Tensor,
        post: Dict[str, List[Tensor]],
        prior: Dict[str, List[Tensor]],
        state: Dict[str, Tensor],
        idxs: List[int],
        use_network_action: bool = False,
    ):
        """
        Forward world model on batch of data using observation/action steps.

        :param path_length: int
        :param action: (batch_size, path_length, action_dim)
        :param embed: (batch_size, path_length, embedding_size)
        :param post: Dict
            mean: (batch_size, path_length, stoch_size)
            std: (batch_size, path_length, stoch_size)
            stoch: (batch_size, path_length, stoch_size)
            deter: (batch_size, path_length, deter_size)
        :param prior: Dict
            mean: (batch_size, path_length, stoch_size)
            std: (batch_size, path_length, stoch_size)
            stoch: (batch_size, path_length, stoch_size)
            deter: (batch_size, path_length, deter_size)
        :param state: Dict
            mean: (batch_size, stoch_size)
            std: (batch_size, stoch_size)
            stoch: (batch_size, stoch_size)
            deter: (batch_size, deter_size)
        :param idxs: List[int] defines which steps for which to use obs.
        :param use_network_action: bool defines whether to use network action or not.

        :return: List[Dict]
            mean: (batch_size, path_length, stoch_size)
            std: (batch_size, path_length, stoch_size)
            stoch: (batch_size, path_length, stoch_size)
            deter: (batch_size, path_length, deter_size)
        """
        actions = []
        ctr = 0
        for step in range(path_length):
            if step == 0:
                # action corresponding to reset obs should always be 0.
                action_to_apply = action[1][:, 0] * 0
            else:
                inp = torch.cat(
                    [action[0][:, step], self.get_features(state).detach()], dim=1
                )
                action_pred = self.primitive_model(inp)
                actions.append(action_pred.unsqueeze(1))
                if use_network_action:
                    action_to_apply = action_pred
                else:
                    action_to_apply = action[1][:, step]
            if step not in idxs:
                prior_params = self.action_step(state, action_to_apply.detach())
                post_params = prior_params
            else:
                (post_params, prior_params,) = self.obs_step(
                    state,
                    action_to_apply.detach(),
                    embed[:, ctr],
                )
                ctr += 1

            for key in post.keys():
                post[key].append(post_params[key].unsqueeze(1))

            for key in prior.keys():
                if step == 0:
                    # Auto encode first action.
                    prior[key].append(post_params[key].unsqueeze(1))
                else:
                    prior[key].append(prior_params[key].unsqueeze(1))
            state = post_params
        return post, prior, torch.cat(actions, dim=1)

    def forward(
        self,
        obs,
        action,
        use_network_action=False,
        state=None,
        batch_indices=None,
        raps_obs_indices=None,
    ):
        """
        Forward world model on trajectory.

        :param obs: (batch_size, path_length, obs_dim)
        :param action: List [(batch_size, path_length, high_level_action_dim), (batch_size, path_length, low_level_action_dim)]
        :param use_network_action:
        :param state:
        :param batch_indices:
        :param raps_obs_indices:

        :return post: Dict
            mean: (batch_size, path_length, stoch_size)
            std: (batch_size, path_length, stoch_size)
            stoch: (batch_size, path_length, stoch_size)
            deter: (batch_size, path_length, deter_size)
        :return prior:
            mean: (batch_size, path_length, stoch_size)
            std: (batch_size, path_length, stoch_size)
            stoch: (batch_size, path_length, stoch_size)
            deter: (batch_size, path_length, deter_size)
        :return post_dist:
            mean: (batch_size*(path_length), stoch_size)
            std: (batch_size*(path_length), stoch_size)
        :return prior_dist:
            mean: (batch_size*(path_length), stoch_size)
            std: (batch_size*(path_length), stoch_size)
        :return image_dist:
            mean: (batch_size*(path_length), obs_dim)
        :return reward_dist:
            mean: (batch_size*(raps_path_length), 1)
        :return pred_discount_dist
            logits: (batch_size*(raps_path_length), 1)
        :return embed: (batch_size, path_length, embed_dim)
        :return low_level_action_preds: (batch_size, path_length, low_level_action_dim)
        """
        assert (
            obs.shape[:2] == action[0].shape[:2] == action[1].shape[:2]
        ), "Obs and action first two dimensions should be the same."
        original_batch_size = action[1].shape[0]
        path_length = action[1].shape[1]
        if state is None:
            state = self.initial(original_batch_size)
        post, prior = (
            dict(mean=[], std=[], stoch=[], deter=[]),
            dict(mean=[], std=[], stoch=[], deter=[]),
        )
        obs_path_len = obs.shape[1]
        obs = obs.reshape(-1, obs.shape[-1])
        embed = self.encode(obs)
        embedding_size = embed.shape[1]
        embed = embed.reshape(original_batch_size, obs_path_len, embedding_size)

        if obs_path_len < path_length:
            idxs = raps_obs_indices.tolist()
        else:
            idxs = np.arange(
                0,
                path_length,
                1,
            ).tolist()

        post, prior, low_level_action_preds = self.forward_batch(
            path_length,
            action,
            embed,
            post,
            prior,
            state,
            idxs,
            use_network_action,
        )

        for key in post.keys():
            post[key] = torch.cat(post[key], dim=1)

        for key in prior.keys():
            prior[key] = torch.cat(prior[key], dim=1)

        if self.use_prior_instead_of_posterior:
            # in this case, o_hat_t depends on a_t-1 and o_t-1, reset obs decoded from null state + action
            # only works when first state is reset obs and never changes
            feat = self.get_features(prior)
        else:
            feat = self.get_features(post)

        raps_obs_feat = feat[:, raps_obs_indices]
        raps_obs_feat = raps_obs_feat.reshape(-1, raps_obs_feat.shape[-1])

        if batch_indices.shape != raps_obs_indices.shape:
            feat = get_indexed_arr_from_batch_indices(feat, batch_indices).reshape(
                -1, feat.shape[-1]
            )
        else:
            feat = feat[:, batch_indices]

        images = self.decode(feat)
        rewards = self.reward(raps_obs_feat)
        pred_discounts = self.pred_discount(raps_obs_feat)

        if batch_indices.shape != raps_obs_indices.shape:
            post_dist = self.get_dist(
                get_indexed_arr_from_batch_indices(post["mean"], batch_indices).reshape(
                    -1, post["mean"].shape[-1]
                ),
                get_indexed_arr_from_batch_indices(post["std"], batch_indices).reshape(
                    -1, post["std"].shape[-1]
                ),
            )
            prior_dist = self.get_dist(
                get_indexed_arr_from_batch_indices(
                    prior["mean"], batch_indices
                ).reshape(-1, prior["mean"].shape[-1]),
                get_indexed_arr_from_batch_indices(prior["std"], batch_indices).reshape(
                    -1, prior["std"].shape[-1]
                ),
            )
        else:
            post_dist = self.get_dist(
                post["mean"][:, batch_indices].reshape(-1, post["mean"].shape[-1]),
                post["std"][:, batch_indices].reshape(-1, post["std"].shape[-1]),
            )
            prior_dist = self.get_dist(
                prior["mean"][:, batch_indices].reshape(-1, prior["mean"].shape[-1]),
                prior["std"][:, batch_indices].reshape(-1, prior["std"].shape[-1]),
            )
        image_dist = self.get_dist(images, ptu.ones_like(images), dims=3)
        if self.reward_classifier:
            reward_dist = self.get_dist(rewards, None, normal=False)
        else:
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
            low_level_action_preds,
        )

    @jit.script_method
    def forward_high_level_step(
        self,
        state: Dict[str, torch.Tensor],
        observation: torch.Tensor,
        low_level_action: torch.Tensor,
        num_low_level_actions_per_primitive: int,
        high_level_action: torch.Tensor,
        use_raps_obs: bool = False,
        use_true_actions: bool = True,
        use_obs: bool = True,
    ):
        """
        Forward world model over one high level action.

        :param state: Dict
            mean: (batch_size, stoch_size)
            std: (batch_size, stoch_size)
            stoch: (batch_size, stoch_size)
            deter: (batch_size, deter_size)
        :param observation: (batch_size, obs_dim)
        :param low_level_action: (batch_size, low_level_action_dim)
        :param num_low_level_actions_per_primitive: int defines subsampling rate for LL_RAPS
        :param high_level_action: (batch_size, high_level_action_dim)
        :param use_raps_obs:
        :param use_true_actions: if true, use environment low level actions, otherwise use primitive model predictions
        :param use_obs: if true, take obs_steps, otherwise take action_steps
        """
        low_level_action_preds = []
        new_state = state
        for idx in range(0, num_low_level_actions_per_primitive):
            phase = (
                torch.ones((high_level_action.shape[0], 1), device=ptu.device)
                * (idx + 1)
                / num_low_level_actions_per_primitive
            )
            hl = torch.cat((high_level_action, phase), 1)
            inp = torch.cat(
                [hl, self.get_features(new_state)],
                dim=1,
            )
            low_level_action_pred = self.primitive_model(inp)
            if use_obs:
                embed = self.encode(observation[:, idx])
                if use_raps_obs:
                    if idx == num_low_level_actions_per_primitive - 1:
                        if use_true_actions:
                            new_state, _ = self.obs_step(
                                new_state, low_level_action[:, idx], embed
                            )
                        else:
                            new_state, _ = self.obs_step(
                                new_state, low_level_action_pred, embed
                            )
                    else:
                        if use_true_actions:
                            new_state = self.action_step(
                                new_state, low_level_action[:, idx]
                            )
                        else:
                            new_state = self.action_step(
                                new_state, low_level_action_pred
                            )
                else:
                    if use_true_actions:
                        new_state, _ = self.obs_step(
                            new_state, low_level_action[:, idx], embed
                        )
                    else:
                        new_state, _ = self.obs_step(
                            new_state, low_level_action_pred, embed
                        )
            else:
                if use_true_actions:
                    new_state = self.action_step(new_state, low_level_action[:, idx])
                else:
                    new_state = self.action_step(new_state, low_level_action_pred)
            low_level_action_preds.append(low_level_action_pred)
        return new_state, low_level_action_preds


class StateConcatObsWorldModel(WorldModel):
    def __init__(self, *args, vec_obs_size=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.vec_obs_size = vec_obs_size

    @jit.script_method
    def encode(self, obs):
        obs_im = obs[:, : -self.vec_obs_size]
        obs_state = obs[:, -self.vec_obs_size :]
        encoded_im = self.conv_encoder(self.preprocess(obs_im))
        return torch.cat((encoded_im, obs_state), dim=1)


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
    def compute_layernorm_stats(self, input_):
        mu = input_.mean(-1, keepdim=True)
        sigma = input_.std(-1, keepdim=True, unbiased=False)
        return mu, sigma

    @jit.script_method
    def forward(self, input_):
        mu, sigma = self.compute_layernorm_stats(input_)
        return (input_ - mu) / sigma * self.weight + self.bias


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

    def check_forward_input(self, input_: Tensor) -> None:
        if input_.size(1) != self.input_size:
            raise RuntimeError(
                f"input has inconsistent input_size: got {input.size(1)}, expected {self.input_size}"
            )

    def check_forward_hidden(
        self, input_: Tensor, hx: Tensor, hidden_label: str = ""
    ) -> None:
        if input_.size(0) != hx.size(0):
            raise RuntimeError(
                f"Input batch size {input.size(0)} doesn't match hidden {hidden_label} batch size {hx.size(0)}"
            )

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                f"hidden {hidden_label} has inconsistent hidden_size: got {hx.size(1)}, expected {self.hidden_size}"
            )

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)
