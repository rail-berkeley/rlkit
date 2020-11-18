import torch
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.model_based.dreamer.conv_networks import CNN, DCNN
from rlkit.torch.model_based.dreamer.mlp import Mlp


class WorldModel(PyTorchModule):
    def __init__(
        self,
        action_dim,
        stochastic_state_size=60,
        deterministic_state_size=400,
        embedding_size=1024,
        rssm_hidden_size=400,
        model_hidden_size=400,
        model_act=F.elu,
        depth=32,
        conv_act=F.relu,
        reward_num_layers=2,
        pcont_num_layers=3,
    ):
        super().__init__()
        self.obs_step_mlp = Mlp(
            hidden_sizes=[rssm_hidden_size],
            input_size=embedding_size + deterministic_state_size,
            output_size=2 * stochastic_state_size,
            hidden_activation=model_act,
            hidden_init=torch.nn.init.xavier_uniform_,
        )
        self.img_step_layer = torch.nn.Linear(
            stochastic_state_size + action_dim, deterministic_state_size
        )
        torch.nn.init.xavier_uniform_(self.img_step_layer.weight)
        self.img_step_layer.bias.data.fill_(0)
        self.img_step_mlp = Mlp(
            hidden_sizes=[rssm_hidden_size],
            input_size=deterministic_state_size,
            output_size=2 * stochastic_state_size,
            hidden_activation=model_act,
            hidden_init=torch.nn.init.xavier_uniform_,
        )
        self.rnn = torch.nn.GRUCell(deterministic_state_size, deterministic_state_size)
        self.conv_encoder = CNN(
            64,
            64,
            3,
            [4] * 4,
            [depth, depth * 2, depth * 4, depth * 8],
            [2] * 4,
            [0] * 4,
            hidden_activation=conv_act,
            hidden_init=torch.nn.init.xavier_uniform_,
        )

        self.conv_decoder = DCNN(
            stochastic_state_size + deterministic_state_size,
            [],
            1,
            1,
            depth * 32,
            6,
            2,
            3,
            [5, 5, 6],
            [depth * 4, depth * 2, depth * 1],
            [2] * 3,
            [0] * 3,
            hidden_activation=conv_act,
            hidden_init=torch.nn.init.xavier_uniform_,
        )

        self.reward = Mlp(
            hidden_sizes=[model_hidden_size] * reward_num_layers,
            input_size=stochastic_state_size + deterministic_state_size,
            output_size=1,
            hidden_activation=model_act,
            hidden_init=torch.nn.init.xavier_uniform_,
        )
        self.pcont = Mlp(
            hidden_sizes=[model_hidden_size] * pcont_num_layers,
            input_size=stochastic_state_size + deterministic_state_size,
            output_size=1,
            hidden_activation=model_act,
            hidden_init=torch.nn.init.xavier_uniform_,
        )

        self.model_act = model_act
        self.feature_size = stochastic_state_size + deterministic_state_size
        self.stochastic_state_size = stochastic_state_size
        self.deterministic_state_size = deterministic_state_size
        self.modules = [
            self.obs_step_mlp,
            self.img_step_layer,
            self.img_step_mlp,
            self.rnn,
            self.conv_encoder,
            self.conv_decoder,
            self.reward,
            self.pcont,
        ]

    def obs_step(self, prev_state, prev_action, embed):
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        x = self.obs_step_mlp(x)
        mean, std = x.split(self.stochastic_state_size, -1)
        std = F.softplus(std) + 0.1
        stoch = self.get_dist(mean, std).rsample()
        post = {"mean": mean, "std": std, "stoch": stoch, "deter": prior["deter"]}
        return post, prior

    def img_step(self, prev_state, prev_action):
        x = torch.cat([prev_state["stoch"], prev_action], -1)
        x = self.img_step_layer(x)
        x = self.model_act(x)
        deter_new = self.rnn(x, prev_state["deter"])
        x = self.img_step_mlp(deter_new)
        mean, std = x.split(self.stochastic_state_size, -1)
        std = F.softplus(std) + 0.1
        stoch = self.get_dist(mean, std).rsample()
        prior = {"mean": mean, "std": std, "stoch": stoch, "deter": deter_new}
        return prior

    def forward_batch(
        self,
        obs,
        action,
        state=None,
    ):
        embed = self.encode(obs)
        post_params, prior_params = self.obs_step(state, action, embed)
        feat = self.get_feat(post_params)
        image_params = self.decode(feat)
        reward_params = self.reward(feat)
        pcont_params = self.pcont(feat)
        return (
            post_params,
            prior_params,
            image_params,
            reward_params,
            pcont_params,
            embed,
        )

    def forward(self, obs, action):
        original_batch_size = obs.shape[0]
        state = self.initial(original_batch_size)
        path_length = obs.shape[1]
        post, prior = dict(mean=[], std=[], stoch=[], deter=[]), dict(
            mean=[], std=[], stoch=[], deter=[]
        )
        images, rewards, pconts, embeds = [], [], [], []

        for i in range(path_length):
            (
                post_params,
                prior_params,
                image_params,
                reward_params,
                pcont_params,
                embed,
            ) = self.forward_batch(obs[:, i], action[:, i], state)
            images.append(image_params)
            rewards.append(reward_params)
            pconts.append(pcont_params)
            embeds.append(embed)
            for k in post.keys():
                post[k].append(post_params[k].unsqueeze(1))

            for k in prior.keys():
                prior[k].append(prior_params[k].unsqueeze(1))
            state = post_params

        images = torch.cat(images)
        rewards = torch.cat(rewards)
        pconts = torch.cat(pconts)
        embeds = torch.cat(embeds)
        for k in post.keys():
            post[k] = torch.cat(post[k], dim=1)

        for k in prior.keys():
            prior[k] = torch.cat(prior[k], dim=1)

        post_dist = self.get_dist(post["mean"], post["std"])
        prior_dist = self.get_dist(prior["mean"], prior["std"])
        image_dist = self.get_dist(images, ptu.ones_like(images), dims=3)
        reward_dist = self.get_dist(rewards, ptu.ones_like(rewards))
        pcont_dist = self.get_dist(pconts, None, normal=False)
        return (
            post,
            prior,
            post_dist,
            prior_dist,
            image_dist,
            reward_dist,
            pcont_dist,
            embeds,
        )

    def get_feat(self, state):
        return torch.cat([state["stoch"], state["deter"]], -1)

    def get_dist(self, mean, std, dims=1, normal=True):
        if normal:
            return torch.distributions.Independent(
                torch.distributions.Normal(mean, std), dims
            )
        else:
            return torch.distributions.Independent(
                torch.distributions.Bernoulli(logits=mean), dims
            )

    def encode(self, obs):
        return self.conv_encoder(self.preprocess(obs))

    def decode(self, feat):
        return self.conv_decoder(feat)

    def preprocess(self, obs):
        obs = obs / 255.0 - 0.5
        return obs

    def initial(self, batch_size):
        return dict(
            mean=ptu.zeros([batch_size, self.stochastic_state_size]),
            std=ptu.zeros([batch_size, self.stochastic_state_size]),
            stoch=ptu.zeros([batch_size, self.stochastic_state_size]),
            deter=ptu.zeros([batch_size, self.deterministic_state_size]),
        )


class MultitaskWorldModel(WorldModel):
    def preprocess(self, obs):
        image_obs, one_hots = (
            obs[:, : 64 * 64 * 3],
            obs[:, 64 * 64 * 3 :],
        )
        image_obs = super(MultitaskWorldModel, self).preprocess(image_obs)
        return image_obs

    def encode(self, obs):
        image_obs, one_hots = (
            obs[:, : 64 * 64 * 3],
            obs[:, 64 * 64 * 3 :],
        )
        image_obs = self.preprocess(obs)
        encoded_obs = self.conv_encoder(image_obs)
        latent = torch.cat((encoded_obs, one_hots), dim=1)
        return latent