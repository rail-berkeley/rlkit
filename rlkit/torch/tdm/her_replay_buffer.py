import numpy as np

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer


class HerReplayBuffer(EnvReplayBuffer):
    """
    Save goals from the same trajectory into the replay buffer.
    Only add_path is implemented.

    Implementation details:
     - Every sample from [0, self._size] will be valid.
    """
    def __init__(
            self,
            max_size,
            env,
            num_goals_to_sample=4,
            fraction_goals_are_rollout_goals=None,
    ):
        """

        :param max_size:
        :param observation_dim:
        :param action_dim:
        :param num_goals_to_sample:
        :param fraction_goals_are_rollout_goals:
        """
        super().__init__(max_size, env)
        self.num_goals_to_sample = num_goals_to_sample
        self._goals = np.zeros((max_size, self.env.goal_dim))
        self._num_steps_left = np.zeros((max_size, 1))
        if fraction_goals_are_rollout_goals is None:
            fraction_goals_are_rollout_goals = (
                    1. / num_goals_to_sample
            )
        self.fraction_goals_are_rollout_goals = (
            fraction_goals_are_rollout_goals
        )

        # Let j be any index in self._idx_to_future_obs_idx[i]
        # Then self._next_obs[j] is a valid next observation for observation i
        self._idx_to_future_obs_idx = [None] * max_size

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        goals = path["goals"]
        num_steps_left = path["num_steps_left"]
        path_len = len(rewards)

        actions = flatten_n(actions)
        obs = flatten_n(obs)
        next_obs = flatten_n(next_obs)

        if self._top + path_len >= self._max_replay_buffer_size:
            num_pre_wrap_steps = self._max_replay_buffer_size - self._top
            # numpy slice
            pre_wrap_buffer_slice = np.s_[
                                    self._top:self._top + num_pre_wrap_steps, :
                                    ]
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, :]

            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._observations[buffer_slice] = obs[path_slice]
                self._actions[buffer_slice] = actions[path_slice]
                self._rewards[buffer_slice] = rewards[path_slice]
                self._next_obs[buffer_slice] = next_obs[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                self._goals[buffer_slice] = goals[path_slice]
                self._num_steps_left[buffer_slice] = num_steps_left[path_slice]
            # Pointers from before the wrap
            for i in range(self._top, self._max_replay_buffer_size):
                self._idx_to_future_obs_idx[i] = np.hstack((
                    # Pre-wrap indices
                    np.arange(i, self._max_replay_buffer_size),
                    # Post-wrap indices
                    np.arange(0, num_post_wrap_steps)
                ))
            # Pointers after the wrap
            for i in range(0, num_post_wrap_steps):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i,
                    num_post_wrap_steps,
                )
        else:
            slc = np.s_[self._top:self._top + path_len, :]
            self._observations[slc] = obs
            self._actions[slc] = actions
            self._rewards[slc] = rewards
            self._next_obs[slc] = next_obs
            self._terminals[slc] = terminals
            self._goals[slc] = goals
            self._num_steps_left[slc] = num_steps_left
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i, self._top + path_len
                )
        self._top = (self._top + path_len) % self._max_replay_buffer_size
        self._size = min(self._size + path_len, self._max_replay_buffer_size)

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        next_obs_idxs = []
        for i in indices:
            possible_next_obs_idxs = self._idx_to_future_obs_idx[i]
            # This is generally faster than random.choice. Makes you wonder what
            # random.choice is doing
            num_options = len(possible_next_obs_idxs)
            if num_options == 1:
                next_obs_i = 0
            else:
                next_obs_i = int(np.random.randint(0, num_options))
            next_obs_idxs.append(possible_next_obs_idxs[next_obs_i])
        next_obs_idxs = np.array(next_obs_idxs)
        resampled_goals = self.env.convert_obs_to_goals(
            self._next_obs[next_obs_idxs]
        )
        num_goals_are_from_rollout = int(
            batch_size * self.fraction_goals_are_rollout_goals
        )
        if num_goals_are_from_rollout > 0:
            resampled_goals[:num_goals_are_from_rollout] = self._goals[
                indices[:num_goals_are_from_rollout]
            ]
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            goals_used_for_rollout=self._goals[indices],
            resampled_goals=resampled_goals,
            num_steps_left=self._num_steps_left[indices],
        )


def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))
