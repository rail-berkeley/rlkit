import gc
import os
import pickle
import warnings

import h5py
import numpy as np

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim
from rlkit.torch.model_based.dreamer.utils import (
    get_batch_length_indices,
    get_indexed_arr_from_batch_indices,
)


class EpisodeReplayBuffer(SimpleReplayBuffer):
    def __init__(
        self,
        n_envs,
        observation_dim,
        action_dim,
        max_replay_buffer_size,
        max_path_length,
        replace=True,
        batch_length=50,
        use_batch_length=False,
    ):
        self.n_envs = n_envs

        self.max_path_length = max_path_length
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros(
            (max_replay_buffer_size, max_path_length + 1, observation_dim),
            dtype=np.uint8,
        )
        self._actions = np.zeros(
            (max_replay_buffer_size, max_path_length + 1, action_dim)
        )
        self._rewards = np.zeros((max_replay_buffer_size, max_path_length + 1, 1))
        self._terminals = np.zeros(
            (max_replay_buffer_size, max_path_length + 1, 1), dtype="uint8"
        )
        self._replace = replace
        self.batch_length = batch_length
        self.use_batch_length = use_batch_length
        self._top = 0
        self._size = 0

    def add_path(self, path):
        # transpose to change from path collector format (path_length, batch) or buffer format (batch, path_length)
        self._observations[self._top : self._top + self.n_envs] = path[
            "observations"
        ].transpose(1, 0, 2)
        self._actions[self._top : self._top + self.n_envs] = path["actions"].transpose(
            1, 0, 2
        )
        self._rewards[self._top : self._top + self.n_envs] = np.expand_dims(
            path["rewards"].transpose(1, 0), -1
        )
        self._terminals[self._top : self._top + self.n_envs] = np.expand_dims(
            path["terminals"].transpose(1, 0), -1
        )

        self._advance()

    def _advance(self):
        self._top = (self._top + self.n_envs) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += self.n_envs

    def random_batch(self, batch_size):
        if self.use_batch_length:
            indices = np.random.choice(
                self._size,
                size=batch_size,
                replace=self._replace or self._size < batch_size,
            )
            if not self._replace and self._size < batch_size:
                warnings.warn(
                    "Replace was set to false, but is temporarily set to true \
                    because batch size is larger than current size of replay."
                )
            batch_indices = get_batch_length_indices(
                self.max_path_length, self.batch_length, batch_size
            )
            observations = get_indexed_arr_from_batch_indices(
                self._observations[indices], batch_indices
            )
            actions = get_indexed_arr_from_batch_indices(
                self._actions[indices], batch_indices
            )
            rewards = get_indexed_arr_from_batch_indices(
                self._rewards[indices], batch_indices
            )
            terminals = get_indexed_arr_from_batch_indices(
                self._terminals[indices], batch_indices
            )
        else:
            indices = np.random.choice(
                self._size,
                size=batch_size,
                replace=self._replace or self._size < batch_size,
            )
            if not self._replace and self._size < batch_size:
                warnings.warn(
                    "Replace was set to false, but is temporarily set to true \
                    because batch size is larger than current size of replay."
                )
            observations = self._observations[indices]
            actions = self._actions[indices]
            rewards = self._rewards[indices]
            terminals = self._terminals[indices]
        assert (
            observations.shape[0]
            == actions.shape[0]
            == rewards.shape[0]
            == terminals.shape[0]
            == batch_size
        )
        batch = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )
        return batch

    def get_diagnostics(self):
        d = super().get_diagnostics()
        d["reward_in_buffer"] = self._rewards.sum()
        return d

    def save(self, path, suffix):
        observations = self._observations
        actions = self._actions
        rewards = self._rewards
        terminals = self._terminals

        delattr(self, "_observations")
        delattr(self, "_actions")
        delattr(self, "_rewards")
        delattr(self, "_terminals")

        pickle.dump(self, open(os.path.join(path, suffix), "wb"))

        base_suffix = suffix.replace(".pkl", "")
        f = h5py.File(os.path.join(path, base_suffix + "_contents.hdf5"), "w")
        f.create_dataset(
            "observations", data=observations, compression="gzip", compression_opts=9
        )
        f.create_dataset(
            "actions", data=actions, compression="gzip", compression_opts=9
        )
        f.create_dataset(
            "rewards", data=rewards, compression="gzip", compression_opts=9
        )
        f.create_dataset(
            "terminals", data=terminals, compression="gzip", compression_opts=9
        )
        f.close()

        self._observations = observations
        self._actions = actions
        self._rewards = rewards
        self._terminals = terminals

    def load(self, path, suffix):
        replay_buffer = pickle.load(open(os.path.join(path, suffix), "rb"))

        base_suffix = suffix.replace(".pkl", "")
        f = h5py.File(os.path.join(path, base_suffix + "buffer_contents.hdf5"), "r")
        replay_buffer._observations = f["observations"][:]
        replay_buffer._actions = f["actions"][:]
        replay_buffer._rewards = f["rewards"][:]
        replay_buffer._terminals = f["terminals"][:]
        f.close()

        return replay_buffer


class EpisodeReplayBufferLowLevelRAPS(EpisodeReplayBuffer):
    def __init__(
        self,
        n_envs,
        observation_dim,
        action_dim,
        max_replay_buffer_size,
        max_path_length,
        num_low_level_actions_per_primitive,
        low_level_action_dim,
        replace=True,
        batch_length=50,
        use_batch_length=False,
        prioritize_fraction=0,
        uniform_priorities=True,
    ):
        self.n_envs = n_envs

        self.max_path_length = max_path_length
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros(
            (
                max_replay_buffer_size,
                max_path_length * num_low_level_actions_per_primitive + 1,
                observation_dim,
            ),
            dtype=np.uint8,
        )
        self._low_level_actions = np.zeros(
            (
                max_replay_buffer_size,
                max_path_length * num_low_level_actions_per_primitive + 1,
                low_level_action_dim,
            )
        )
        self._high_level_actions = np.zeros(
            (
                max_replay_buffer_size,
                max_path_length * num_low_level_actions_per_primitive + 1,
                action_dim + 1,
            )
        )
        self._rewards = np.zeros((max_replay_buffer_size, max_path_length + 1, 1))
        self._terminals = np.zeros(
            (max_replay_buffer_size, max_path_length + 1, 1), dtype="uint8"
        )
        self._replace = replace
        self.batch_length = batch_length
        self.use_batch_length = use_batch_length
        self._top = 0
        self._size = 0
        self.prioritize_fraction = prioritize_fraction
        self.uniform_priorities = uniform_priorities

    def add_path(self, path):
        self._observations[self._top : self._top + self.n_envs] = path["observations"]
        self._low_level_actions[self._top : self._top + self.n_envs] = path[
            "low_level_actions"
        ]
        self._high_level_actions[self._top : self._top + self.n_envs] = path[
            "high_level_actions"
        ]
        self._rewards[self._top : self._top + self.n_envs] = np.expand_dims(
            path["rewards"].transpose(1, 0), -1
        )
        self._terminals[self._top : self._top + self.n_envs] = np.expand_dims(
            path["terminals"].transpose(1, 0), -1
        )

        self._advance()

    def _advance(self):
        self._top = (self._top + self.n_envs) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += self.n_envs

    def random_batch(self, batch_size):
        mask = np.where(self._rewards[: self._size].sum(axis=1) > 0, 1, 0)[:, 0]
        prioritized_indices = np.array(range(self._size))[mask > 0]
        if prioritized_indices.shape[0] > 0 and self.prioritize_fraction > 0:
            if self.uniform_priorities:
                priorities = np.ones_like(prioritized_indices)
            else:
                priorities = self._rewards[: self._size][:, :, 0][
                    prioritized_indices
                ].sum(axis=1)
            normalized_priorities = priorities / priorities.sum()
            indices = np.random.choice(
                prioritized_indices.shape[0],
                size=int(self.prioritize_fraction * batch_size),
                replace=prioritized_indices.shape[0]
                < int(self.prioritize_fraction * batch_size),
                p=normalized_priorities,
            )
            prioritized_indices = prioritized_indices[indices]
            indices = np.random.choice(
                self._size,
                size=batch_size - indices.shape[0],
                replace=self._replace or self._size < batch_size,
            )
            indices = np.concatenate((indices, prioritized_indices))
        else:
            indices = np.random.choice(
                self._size,
                size=batch_size,
                replace=self._replace or self._size < batch_size,
            )

        observations = self._observations[indices]
        high_level_actions = self._high_level_actions[indices]
        low_level_actions = self._low_level_actions[indices]
        rewards = self._rewards[indices]
        terminals = self._terminals[indices]
        batch = dict(
            observations=observations,
            high_level_actions=high_level_actions,
            low_level_actions=low_level_actions,
            rewards=rewards,
            terminals=terminals,
        )
        return batch

    def load_buffer(self, filename, num_primitives):
        print("LOADING REPLAY BUFFER")
        with h5py.File(filename, "r") as f:
            observations = np.array(f["observations"][:])
            low_level_actions = np.array(f["low_level_actions"][:])
            high_level_actions = np.array(f["high_level_actions"][:])
            rewards = np.array(f["rewards"][:])
            terminals = np.array(f["terminals"][:])
        num_trajs = observations.shape[0]
        self._observations[:num_trajs] = observations
        self._low_level_actions[:num_trajs] = low_level_actions
        argmax = np.argmax(high_level_actions[:, :, :num_primitives], axis=-1)
        one_hots = np.eye(num_primitives)[argmax]
        one_hots[:, 0:1, :] = np.zeros((one_hots.shape[0], 1, num_primitives))
        high_level_actions = np.concatenate(
            (one_hots, high_level_actions[:, :, num_primitives:]),
            axis=-1,
        )
        self._high_level_actions[:num_trajs] = high_level_actions
        self._rewards[:num_trajs] = rewards
        self._terminals[:num_trajs] = terminals
        self._top = num_trajs
        self._size = num_trajs

        del observations
        del low_level_actions
        del high_level_actions
        del rewards
        del terminals
        gc.collect()

    def save(self, path, suffix):
        observations = self._observations
        low_level_actions = self._low_level_actions
        high_level_actions = self._high_level_actions
        rewards = self._rewards
        terminals = self._terminals

        delattr(self, "_observations")
        delattr(self, "_low_level_actions")
        delattr(self, "_high_level_actions")
        delattr(self, "_rewards")
        delattr(self, "_terminals")

        pickle.dump(self, open(os.path.join(path, suffix), "wb"))

        base_suffix = suffix.replace(".pkl", "")
        f = h5py.File(os.path.join(path, base_suffix + "_contents.hdf5"), "w")
        f.create_dataset(
            "observations", data=observations, compression="gzip", compression_opts=9
        )
        f.create_dataset(
            "low_level_actions",
            data=low_level_actions,
            compression="gzip",
            compression_opts=9,
        )
        f.create_dataset(
            "high_level_actions",
            data=high_level_actions,
            compression="gzip",
            compression_opts=9,
        )
        f.create_dataset(
            "rewards", data=rewards, compression="gzip", compression_opts=9
        )
        f.create_dataset(
            "terminals", data=terminals, compression="gzip", compression_opts=9
        )
        f.close()

        self._observations = observations
        self._low_level_actions = low_level_actions
        self._high_level_actions = high_level_actions
        self._rewards = rewards
        self._terminals = terminals

    def load(self, path, suffix):
        replay_buffer = pickle.load(open(os.path.join(path, suffix), "rb"))

        base_suffix = suffix.replace(".pkl", "")
        f = h5py.File(os.path.join(path, base_suffix + "buffer_contents.hdf5"), "r")
        replay_buffer._observations = f["observations"][:]
        replay_buffer._low_level_actions = f["low_level_actions"][:]
        replay_buffer._high_level_actions = f["high_level_actions"][:]
        replay_buffer._rewards = f["rewards"][:]
        replay_buffer._terminals = f["terminals"][:]
        f.close()

        return replay_buffer
