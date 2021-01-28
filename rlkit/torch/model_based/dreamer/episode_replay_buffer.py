import warnings

import numpy as np

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim


class EpisodeReplayBuffer(SimpleReplayBuffer):
    def __init__(
        self,
        max_replay_buffer_size,
        env,
        max_path_length,
        observation_dim,
        action_dim,
        replace=True,
    ):
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        self._observation_dim = get_dim(self._ob_space)
        self._action_dim = get_dim(self._action_space)
        self.max_path_length = max_path_length
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros(
            (max_replay_buffer_size, max_path_length, observation_dim),
            dtype=np.uint8,  # todo: figure out what to do in the case of proprioceptive obs
        )
        self._actions = np.zeros((max_replay_buffer_size, max_path_length, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, max_path_length, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros(
            (max_replay_buffer_size, max_path_length, 1), dtype="uint8"
        )
        self._replace = replace

        self._top = 0
        self._size = 0

    def add_path(self, path):
        self._observations[self._top : self._top + self.env.n_envs] = path[
            "observations"
        ].transpose(1, 0, 2)
        self._actions[self._top : self._top + self.env.n_envs] = path[
            "actions"
        ].transpose(1, 0, 2)
        self._rewards[self._top : self._top + self.env.n_envs] = np.expand_dims(
            path["rewards"].transpose(1, 0), -1
        )
        self._terminals[self._top : self._top + self.env.n_envs] = np.expand_dims(
            path["terminals"].transpose(1, 0), -1
        )

        self._advance()

    def _advance(self):
        self._top = (self._top + self.env.n_envs) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += self.env.n_envs

    def random_batch(self, batch_size):
        indices = np.random.choice(
            self._size,
            size=batch_size,
            replace=self._replace or self._size < batch_size,
        )
        if not self._replace and self._size < batch_size:
            warnings.warn(
                "Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay."
            )
        batch_start = np.random.randint(0, self.max_path_length - 50)
        batch = dict(
            observations=self._observations[indices][:, batch_start : batch_start + 50],
            actions=self._actions[indices][:, batch_start : batch_start + 50],
            rewards=self._rewards[indices][:, batch_start : batch_start + 50],
            terminals=self._terminals[indices][:, batch_start : batch_start + 50],
        )
        return batch
