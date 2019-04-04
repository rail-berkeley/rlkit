import numpy as np

import rlkit.torch.pytorch_util as ptu
from multiworld.core.image_env import normalize_image
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.data_management.obs_dict_replay_buffer import flatten_dict
from rlkit.data_management.shared_obs_dict_replay_buffer import \
    SharedObsDictRelabelingBuffer
from rlkit.envs.vae_wrapper import VAEWrappedEnv
from rlkit.torch.vae.vae_trainer import (
    compute_p_x_np_to_np,
    relative_probs_from_log_probs,
)


class OnlineVaeRelabelingBuffer(SharedObsDictRelabelingBuffer):

    def __init__(
            self,
            vae,
            *args,
            decoded_obs_key='image_observation',
            decoded_achieved_goal_key='image_achieved_goal',
            decoded_desired_goal_key='image_desired_goal',
            exploration_rewards_type='None',
            exploration_rewards_scale=1.0,
            vae_priority_type='None',
            start_skew_epoch=0,
            power=1.0,
            internal_keys=None,
            priority_function_kwargs=None,
            relabeling_goal_sampling_mode='vae_prior',
            **kwargs
    ):
        if internal_keys is None:
            internal_keys = []

        for key in [
            decoded_obs_key,
            decoded_achieved_goal_key,
            decoded_desired_goal_key
        ]:
            if key not in internal_keys:
                internal_keys.append(key)
        super().__init__(internal_keys=internal_keys, *args, **kwargs)
        assert isinstance(self.env, VAEWrappedEnv)
        self.vae = vae
        self.decoded_obs_key = decoded_obs_key
        self.decoded_desired_goal_key = decoded_desired_goal_key
        self.decoded_achieved_goal_key = decoded_achieved_goal_key
        self.exploration_rewards_type = exploration_rewards_type
        self.exploration_rewards_scale = exploration_rewards_scale
        self.start_skew_epoch = start_skew_epoch
        self.vae_priority_type = vae_priority_type
        self.power = power
        self._relabeling_goal_sampling_mode = relabeling_goal_sampling_mode

        self._give_explr_reward_bonus = (
                exploration_rewards_type != 'None'
                and exploration_rewards_scale != 0.
        )
        self._exploration_rewards = np.zeros((self.max_size, 1))
        self._prioritize_vae_samples = (
                vae_priority_type != 'None'
                and power != 0.
        )
        self._vae_sample_priorities = np.zeros((self.max_size, 1))
        self._vae_sample_probs = None

        type_to_function = {
            'vae_prob': self.vae_prob,
            'None': self.no_reward,
        }

        self.exploration_reward_func = (
            type_to_function[self.exploration_rewards_type]
        )
        self.vae_prioritization_func = (
            type_to_function[self.vae_priority_type]
        )

        if priority_function_kwargs is None:
            self.priority_function_kwargs = dict()
        else:
            self.priority_function_kwargs = priority_function_kwargs

        self.epoch = 0
        self._register_mp_array("_exploration_rewards")
        self._register_mp_array("_vae_sample_priorities")

    def add_path(self, path):
        self.add_decoded_vae_goals_to_path(path)
        super().add_path(path)

    def add_decoded_vae_goals_to_path(self, path):
        # decoding the self-sampled vae images should be done in batch (here)
        # rather than in the env for efficiency
        desired_goals = flatten_dict(
            path['observations'],
            [self.desired_goal_key]
        )[self.desired_goal_key]
        desired_decoded_goals = self.env._decode(desired_goals)
        desired_decoded_goals = desired_decoded_goals.reshape(
            len(desired_decoded_goals),
            -1
        )
        for idx, next_obs in enumerate(path['observations']):
            path['observations'][idx][self.decoded_desired_goal_key] = \
                desired_decoded_goals[idx]
            path['next_observations'][idx][self.decoded_desired_goal_key] = \
                desired_decoded_goals[idx]

    def get_diagnostics(self):
        if self._vae_sample_probs is None or self._vae_sample_priorities is None:
            stats = create_stats_ordered_dict(
                'VAE Sample Weights',
                np.zeros(self._size),
            )
            stats.update(create_stats_ordered_dict(
                'VAE Sample Probs',
                np.zeros(self._size),
            ))
        else:
            vae_sample_priorities = self._vae_sample_priorities[:self._size]
            vae_sample_probs = self._vae_sample_probs[:self._size]
            stats = create_stats_ordered_dict(
                'VAE Sample Weights',
                vae_sample_priorities,
            )
            stats.update(create_stats_ordered_dict(
                'VAE Sample Probs',
                vae_sample_probs,
            ))
        return stats

    def refresh_latents(self, epoch):
        self.epoch = epoch
        self.skew = (self.epoch > self.start_skew_epoch)
        batch_size = 512
        next_idx = min(batch_size, self._size)

        if self.exploration_rewards_type == 'hash_count':
            # you have to count everything then compute exploration rewards
            cur_idx = 0
            next_idx = min(batch_size, self._size)
            while cur_idx < self._size:
                idxs = np.arange(cur_idx, next_idx)
                normalized_imgs = (
                    normalize_image(self._next_obs[self.decoded_obs_key][idxs])
                )
                cur_idx = next_idx
                next_idx += batch_size
                next_idx = min(next_idx, self._size)

        cur_idx = 0
        obs_sum = np.zeros(self.vae.representation_size)
        obs_square_sum = np.zeros(self.vae.representation_size)
        while cur_idx < self._size:
            idxs = np.arange(cur_idx, next_idx)
            self._obs[self.observation_key][idxs] = \
                self.env._encode(
                    normalize_image(self._obs[self.decoded_obs_key][idxs])
                )
            self._next_obs[self.observation_key][idxs] = \
                self.env._encode(
                    normalize_image(self._next_obs[self.decoded_obs_key][idxs])
                )
            # WARNING: we only refresh the desired/achieved latents for
            # "next_obs". This means that obs[desired/achieve] will be invalid,
            # so make sure there's no code that references this.
            # TODO: enforce this with code and not a comment
            self._next_obs[self.desired_goal_key][idxs] = \
                self.env._encode(
                    normalize_image(self._next_obs[self.decoded_desired_goal_key][idxs])
                )
            self._next_obs[self.achieved_goal_key][idxs] = \
                self.env._encode(
                    normalize_image(self._next_obs[self.decoded_achieved_goal_key][idxs])
                )
            normalized_imgs = (
                normalize_image(self._next_obs[self.decoded_obs_key][idxs])
            )
            if self._give_explr_reward_bonus:
                rewards = self.exploration_reward_func(
                    normalized_imgs,
                    idxs,
                    **self.priority_function_kwargs
                )
                self._exploration_rewards[idxs] = rewards.reshape(-1, 1)
            if self._prioritize_vae_samples:
                if (
                        self.exploration_rewards_type == self.vae_priority_type
                        and self._give_explr_reward_bonus
                ):
                    self._vae_sample_priorities[idxs] = (
                        self._exploration_rewards[idxs]
                    )
                else:
                    self._vae_sample_priorities[idxs] = (
                        self.vae_prioritization_func(
                            normalized_imgs,
                            idxs,
                            **self.priority_function_kwargs
                        ).reshape(-1, 1)
                    )
            obs_sum+= self._obs[self.observation_key][idxs].sum(axis=0)
            obs_square_sum+= np.power(self._obs[self.observation_key][idxs], 2).sum(axis=0)

            cur_idx = next_idx
            next_idx += batch_size
            next_idx = min(next_idx, self._size)
        self.vae.dist_mu = obs_sum/self._size
        self.vae.dist_std = np.sqrt(obs_square_sum/self._size - np.power(self.vae.dist_mu, 2))

        if self._prioritize_vae_samples:
            """
            priority^power is calculated in the priority function
            for image_bernoulli_prob or image_gaussian_inv_prob and
            directly here if not.
            """
            if self.vae_priority_type == 'vae_prob':
                self._vae_sample_priorities[:self._size] = relative_probs_from_log_probs(
                    self._vae_sample_priorities[:self._size]
                )
                self._vae_sample_probs = self._vae_sample_priorities[:self._size]
            else:
                self._vae_sample_probs = self._vae_sample_priorities[:self._size] ** self.power
            p_sum = np.sum(self._vae_sample_probs)
            assert p_sum > 0, "Unnormalized p sum is {}".format(p_sum)
            self._vae_sample_probs /= np.sum(self._vae_sample_probs)
            self._vae_sample_probs = self._vae_sample_probs.flatten()

    def sample_weighted_indices(self, batch_size):
        if (
            self._prioritize_vae_samples and
            self._vae_sample_probs is not None and
            self.skew
        ):
            indices = np.random.choice(
                len(self._vae_sample_probs),
                batch_size,
                p=self._vae_sample_probs,
            )
            assert (
                np.max(self._vae_sample_probs) <= 1 and
                np.min(self._vae_sample_probs) >= 0
            )
        else:
            indices = self._sample_indices(batch_size)
        return indices

    def _sample_goals_from_env(self, batch_size):
        self.env.goal_sampling_mode = self._relabeling_goal_sampling_mode
        return self.env.sample_goals(batch_size)

    def sample_buffer_goals(self, batch_size):
        """
        Samples goals from weighted replay buffer for relabeling or exploration.
        Returns None if replay buffer is empty.

        Example of what might be returned:
        dict(
            image_desired_goals: image_achieved_goals[weighted_indices],
            latent_desired_goals: latent_desired_goals[weighted_indices],
        )
        """
        if self._size == 0:
            return None
        weighted_idxs = self.sample_weighted_indices(
            batch_size,
        )
        next_image_obs = normalize_image(
            self._next_obs[self.decoded_obs_key][weighted_idxs]
        )
        next_latent_obs = self._next_obs[self.achieved_goal_key][weighted_idxs]
        return {
            self.decoded_desired_goal_key:  next_image_obs,
            self.desired_goal_key:          next_latent_obs
        }

    def random_vae_training_data(self, batch_size, epoch):
        # epoch no longer needed. Using self.skew in sample_weighted_indices
        # instead.
        weighted_idxs = self.sample_weighted_indices(
            batch_size,
        )

        next_image_obs = normalize_image(
            self._next_obs[self.decoded_obs_key][weighted_idxs]
        )
        return dict(
            next_obs=ptu.from_numpy(next_image_obs)
        )

    def vae_prob(self, next_vae_obs, indices, **kwargs):
        return compute_p_x_np_to_np(
            self.vae,
            next_vae_obs,
            power=self.power,
            **kwargs
        )

    def no_reward(self, next_vae_obs, indices):
        return np.zeros((len(next_vae_obs), 1))

    def _get_sorted_idx_and_train_weights(self):
        idx_and_weights = zip(range(len(self._vae_sample_probs)),
                              self._vae_sample_probs)
        return sorted(idx_and_weights, key=lambda x: x[1])
