from rlkit.torch.core import PyTorchModule
from rlkit.torch.model_based.dreamer.conv_networks import CNN, DCNN
from rlkit.torch.model_based.dreamer.mlp import Mlp
from torch.distributions import Normal
import torch.nn.functional as F
from torch.distributions.transformed_distribution import TransformedDistribution
import torch
import numpy as np
import rlkit.torch.pytorch_util as ptu


class ActorModel(Mlp):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        discrete_continuous_dist=False,
        discrete_action_dim=0,
        continuous_action_dim=0,
        hidden_activation=F.elu,
        min_std=1e-4,
        init_std=5,
        mean_scale=5,
        **kwargs
    ):
        self.discrete_continuous_dist = discrete_continuous_dist
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim
        if self.discrete_continuous_dist:
            self.output_size = self.discrete_action_dim + self.continuous_action_dim * 2
        else:
            self.output_size = self.continuous_action_dim * 2
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=self.output_size,
            hidden_activation=hidden_activation,
            hidden_init=torch.nn.init.xavier_uniform_,
            **kwargs
        )
        self._min_std = min_std
        self._init_std = ptu.from_numpy(np.array(init_std))
        self._mean_scale = mean_scale
        self.modules = self.fcs + [self.last_fc]

    def forward(self, input):
        raw_init_std = torch.log(torch.exp(self._init_std) - 1)
        h = input
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        last = self.last_fc(h)
        if self.discrete_continuous_dist:
            assert last.shape[1] == self.output_size
            mean, continuous_action_std = (
                last[:, : self.discrete_action_dim + self.continuous_action_dim],
                last[:, self.discrete_action_dim + self.continuous_action_dim :],
            )
            discrete_logits, continuous_action_mean, extra = mean.split(
                self.discrete_action_dim, -1
            )
            continuous_action_mean = torch.cat((continuous_action_mean, extra), -1)

            dist1 = OneHotDist(logits=discrete_logits)

            action_mean = self._mean_scale * torch.tanh(
                continuous_action_mean / self._mean_scale
            )
            action_std = (
                F.softplus(continuous_action_std + raw_init_std) + self._min_std
            )

            dist2 = Normal(action_mean, action_std)
            dist2 = TransformedDistribution(dist2, TanhBijector())
            dist2 = torch.distributions.Independent(dist2, 1)
            dist2 = SampleDist(dist2)
            dist = SplitDist(dist1, dist2)
        else:
            action_mean, action_std_dev = last.split(self.continuous_action_dim, -1)
            action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
            action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std

            dist = Normal(action_mean, action_std)
            dist = TransformedDistribution(dist, TanhBijector())
            dist = torch.distributions.Independent(dist, 1)
            dist = SampleDist(dist)
        return dist


# "atanh", "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True

    @property
    def sign(self):
        return 1.0

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2) - x - F.softplus(-2.0 * x))


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        sample = self._dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = (
            torch.argmax(logprob, dim=0)
            .reshape(1, batch_size, 1)
            .expand(1, batch_size, feature_size)
        )
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def rsample(self):
        return self._dist.rsample()


class OneHotDist:
    def __init__(self, logits=None, probs=None):
        self._dist = torch.distributions.Categorical(logits=logits, probs=probs)
        self._num_classes = self._dist.logits.shape[-1]

    @property
    def name(self):
        return "OneHotDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def log_prob(self, values):
        indices = torch.argmax(values, dim=-1)
        return self._dist.log_prob(indices)

    def mean(self):
        return self._dist.mean

    def mode(self):
        return self._one_hot(torch.argmax(self._dist.probs, dim=-1))

    def rsample(self):
        indices = self._dist.sample()
        sample = self._one_hot(indices).float()
        probs = self._dist.probs
        sample += probs - (probs).detach()  # straight through estimator
        return sample

    def _one_hot(self, indices):
        return F.one_hot(indices, self._num_classes)


class SplitDist:
    def __init__(self, dist1, dist2):
        self._dist1 = dist1
        self._dist2 = dist2

    def rsample(self):
        return torch.cat((self._dist1.rsample(), self._dist2.rsample()), -1)

    def mode(self):
        return torch.cat((self._dist1.mode().float(), self._dist2.mode().float()), -1)


class WorldModel(PyTorchModule):
    def __init__(
        self,
        action_dim,
        stochastic_state_size=60,
        deterministic_state_size=400,
        embedding_size=1024,
        model_hidden_size=400,
        model_act=F.elu,
        depth=32,
        conv_act=F.relu,
        reward_num_layers=2,
        pcont_num_layers=3,
    ):
        super().__init__()
        self.obs_step_mlp = Mlp(
            hidden_sizes=[model_hidden_size],
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
            hidden_sizes=[model_hidden_size],
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
        return post_params, prior_params, image_params, reward_params, pcont_params

    def forward(self, obs, action):
        original_batch_size = obs.shape[0]
        state = self.initial(original_batch_size)
        path_length = obs.shape[1]
        post, prior = dict(mean=[], std=[], stoch=[], deter=[]), dict(
            mean=[], std=[], stoch=[], deter=[]
        )
        images, rewards, pconts = [], [], []

        for i in range(path_length):
            (
                post_params,
                prior_params,
                image_params,
                reward_params,
                pcont_params,
            ) = self.forward_batch(obs[:, i], action[:, i], state)
            images.append(image_params)
            rewards.append(reward_params)
            pconts.append(pcont_params)
            for k in post.keys():
                post[k].append(post_params[k].unsqueeze(1))

            for k in prior.keys():
                prior[k].append(prior_params[k].unsqueeze(1))
            state = post_params

        images = torch.cat(images)
        rewards = torch.cat(rewards)
        pconts = torch.cat(pconts)
        for k in post.keys():
            post[k] = torch.cat(post[k], dim=1)

        for k in prior.keys():
            prior[k] = torch.cat(prior[k], dim=1)

        post_dist = self.get_dist(post["mean"], post["std"])
        prior_dist = self.get_dist(prior["mean"], prior["std"])
        image_dist = self.get_dist(images, ptu.ones_like(images), dims=3)
        reward_dist = self.get_dist(rewards, ptu.ones_like(rewards))
        pcont_dist = self.get_dist(pconts, None, normal=False)
        return post, prior, post_dist, prior_dist, image_dist, reward_dist, pcont_dist

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


# "get_parameters" and "FreezeParameters" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def get_parameters(modules):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    def __init__(self, modules):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
                output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]
