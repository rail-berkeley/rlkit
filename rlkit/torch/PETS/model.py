from torch import nn as nn

from rlkit.torch.networks import FlattenMlp
from rlkit.torch.core import torch_ify

class Model(nn.Module):
    def __init__(self, hidden_sizes, obs_dim, action_dim, num_bootstrap, rew_function=None):
        '''
        Usage:
        model = Model(...)

        next_obs = model(obs, action)
        trajectory = model.unroll(obs, action_sequence)

        TODO: handle the different PETS sampling strategies.
        '''
        self.rew_function = rew_function
        self.predict_reward = self.rew_function is None
        if self.predict_reward:
            self.output_dim = obs_dim + 1
        else:
            self.output_dim = obs_dim

        self._nets = []
        for i in range(num_bootstrap):
            # TODO: figure out what the network architecture should be
            self._nets.append(FlattenMlp(hidden_sizes, self.output_dim, self.input_size))

    def forward(self, obs, action, network_idx=None):
        # TODO: is this the usage I want?
        obs = torch_ify(obs)
        action = torch_ify(action)
        if network_idx is None:
            network = random.choice(self._nets)
        else:
            network = self._nets[network_idx]
        output = network(obs, action)
        if self.predict_reward:
            obs = output[:, :-1]
            reward = output[:, -1]
        else:
            obs = output
            reward = self.rew_function(obs, action)
        return obs, reward

    def unroll(self, obs, action_sequence, network_idx=None):
        '''
        obs: batch_size * obs_dim (Tensor)
        action_sequence: batch_size * timesteps * action_dim (Tensor)
        network_idx: int
        return
        observations: batch_size * timesteps * obs_dim
        rewards: batch_size * timesteps
        '''
        # TODO: handle the PETS sampling options
        obs = torch_ify(obs)
        action_sequence = torch_ify(action_sequence)
        n_timesteps = action_sequence.shape[1]
        obs_output = []
        rew_output = []
        for i in range(n_timesteps):
            next_obs, reward = self.forward(obs, action_sequence[:, i, :], network_idx)
            obs_output.append(next_obs)
            rew_output.append(reward)
        observations = torch.stack(obs_output, dim=1)
        rewards = torch.stack(rew_output, dim=1)
        return observations, rewards
