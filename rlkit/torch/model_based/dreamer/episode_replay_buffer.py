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
        batch_length=50,
        use_batch_length=False,
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
            dtype=np.uint8,
        )
        self._actions = np.zeros((max_replay_buffer_size, max_path_length, action_dim))
        self._rewards = np.zeros((max_replay_buffer_size, max_path_length, 1))
        self._terminals = np.zeros(
            (max_replay_buffer_size, max_path_length, 1), dtype="uint8"
        )
        self._replace = replace
        self.batch_length = batch_length
        self.use_batch_length = use_batch_length
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
        if self.use_batch_length:
            indices = np.random.choice(
                self._size,
                size=batch_size,
                replace=True,
            )
            if not self._replace and self._size < batch_size:
                warnings.warn(
                    "Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay."
                )
            batch_start = np.random.randint(
                0, self.max_path_length - self.batch_length, size=(batch_size)
            )
            batch_indices = np.linspace(
                batch_start,
                batch_start + self.batch_length,
                self.batch_length,
                endpoint=False,
            ).astype(int)

            observations = self._observations[indices][
                np.arange(batch_size), batch_indices
            ].transpose(1, 0, 2)
            actions = self._actions[indices][
                np.arange(batch_size), batch_indices
            ].transpose(1, 0, 2)
            rewards = self._rewards[indices][
                np.arange(batch_size), batch_indices
            ].transpose(1, 0, 2)
            terminals = self._terminals[indices][
                np.arange(batch_size), batch_indices
            ].transpose(1, 0, 2)
        else:
            indices = np.random.choice(
                self._size,
                size=batch_size,
                replace=self._replace or self._size < batch_size,
            )
            if not self._replace and self._size < batch_size:
                warnings.warn(
                    "Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay."
                )
            observations = self._observations[indices]
            actions = self._actions[indices]
            rewards = self._rewards[indices]
            terminals = self._terminals[indices]
        batch = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )
        return batch
