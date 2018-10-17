import numpy as np

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer


class RelabelingReplayBuffer(EnvReplayBuffer):
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
            fraction_goals_are_rollout_goals=1.0, # default, no HER
            fraction_resampled_goals_are_env_goals=0.0, # this many goals are just sampled from environment directly
    ):
        """
        :param resampling_strategy: How to resample states from the rest of
        the trajectory?
        - 'uniform': Sample them uniformly
        - 'truncated_geometric': Used a truncated geometric distribution
        """
        super().__init__(max_size, env)
        self._goals = np.zeros((max_size, self.env.goal_dim))
        self._num_steps_left = np.zeros((max_size, 1))
        self.fraction_goals_are_rollout_goals = fraction_goals_are_rollout_goals
        self.fraction_resampled_goals_are_env_goals = fraction_resampled_goals_are_env_goals

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
        num_steps_left = path["rewards"].copy() # path["num_steps_left"] # irrelevant for non-TDM
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

    def _sample_indices(self, batch_size):
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
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
        # recompute rewards
        new_obs = self._observations[indices]
        new_next_obs = self._next_obs[indices]
        new_actions = self._actions[indices]
        new_rewards = self._rewards[indices].copy() # needs to be recomputed
        random_numbers = np.random.rand(batch_size)
        for i in range(batch_size):
            if random_numbers[i] < self.fraction_resampled_goals_are_env_goals:
                resampled_goals[i, :] = self.env.sample_goal_for_rollout()

            new_reward = self.env.compute_her_reward_np(
                new_obs[i, :],
                new_actions[i, :],
                new_next_obs[i, :],
                resampled_goals[i, :],
            )
            new_rewards[i] = new_reward

        batch = dict(
            observations=new_obs,
            actions=new_actions,
            rewards=new_rewards,
            terminals=self._terminals[indices],
            next_observations=new_next_obs,
            goals_used_for_rollout=self._goals[indices],
            resampled_goals=resampled_goals,
            num_steps_left=self._num_steps_left[indices],
            indices=np.array(indices).reshape(-1, 1),
            goals=resampled_goals,
        )
        return batch

def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))


def flatten_env_info(env_infos, env_info_keys):
# Turns list of env_info dicts into env_info dict of 2D np arrays
    return {
        key: flatten_n(
                [env_infos[i][key] for i in range(len(env_infos))]
        )
        for key in env_info_keys
    }
