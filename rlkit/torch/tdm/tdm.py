from collections import OrderedDict

import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.ddpg.ddpg import DDPG
from rlkit.data_management.path_builder import PathBuilder

from rlkit.torch.tdm.base import RandomUniversalPolicy
from rlkit.torch.tdm.networks import TdmNormalizer
from rlkit.torch.tdm.sampling import (
    MultigoalSimplePathSampler,
    multitask_rollout,
)
from rlkit.torch.torch_rl_algorithm import np_to_pytorch_batch


class TemporalDifferenceModel(DDPG):
    def __init__(
            self,
            env,
            qf,
            exploration_policy,

            policy=None,
            replay_buffer=None,

            max_tau=10,
            vectorized=True,
            goal_weights=None,
            tdm_normalizer: TdmNormalizer=None,
            num_pretrain_paths=0,
            normalize_distance=False,

            **ddpg_kwargs
    ):
        """
        :param max_tau: Maximum tau (planning horizon) to train with.
        :param vectorized: Train the QF in vectorized form?
        :param goal_weights: None or the weights for the different goal
        dimensions. These weights are used to compute the distances to the goal.
        """
        DDPG.__init__(
            self,
            env=env,
            qf=qf,
            policy=policy,
            exploration_policy=exploration_policy,
            replay_buffer=replay_buffer,
            **ddpg_kwargs
            # **ddpg_kwargs,
            # **base_kwargs
        )

        self.max_tau = max_tau
        self.vectorized = vectorized
        self._current_path_goal = None
        self._rollout_tau = np.array([self.max_tau])
        self.goal_weights = goal_weights
        if self.goal_weights is not None:
            # In case they were passed in as (e.g.) tuples or list
            self.goal_weights = np.array(self.goal_weights)
            assert self.goal_weights.size == self.env.goal_dim
        self.tdm_normalizer = tdm_normalizer
        self.num_pretrain_paths = num_pretrain_paths
        self.normalize_distance = normalize_distance

        self.eval_sampler = MultigoalSimplePathSampler(
            env=self.env,
            policy=self.eval_policy,
            max_samples=self.num_steps_per_eval,
            max_path_length=self.max_path_length,
            tau_sampling_function=self._sample_max_tau_for_rollout,
            goal_sampling_function=self._sample_goal_for_rollout,
            cycle_taus_for_rollout=True,
        )
        self.pretrain_obs = None

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = batch['goals']
        num_steps_left = batch['num_steps_left']

        """
        Policy operations.
        """
        policy_actions, pre_tanh_value = self.policy(
            obs, goals, num_steps_left, return_preactivations=True,
        )
        pre_activation_policy_loss = (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        q_output = self.qf(
            observations=obs,
            actions=policy_actions,
            num_steps_left=num_steps_left,
            goals=goals,
        )
        raw_policy_loss = - q_output.mean()
        policy_loss = (
                raw_policy_loss +
                pre_activation_policy_loss * self.policy_pre_activation_weight
        )

        """
        Critic operations.
        """
        next_actions = self.target_policy(
            observations=next_obs,
            goals=goals,
            num_steps_left=num_steps_left-1,
        )
        # speed up computation by not backpropping these gradients
        next_actions.detach()
        target_q_values = self.target_qf(
            observations=next_obs,
            actions=next_actions,
            goals=goals,
            num_steps_left=num_steps_left-1,
        )
        q_target = rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()
        q_pred = self.qf(
            observations=obs,
            actions=actions,
            goals=goals,
            num_steps_left=num_steps_left,
        )
        if self.tdm_normalizer:
            q_pred = self.tdm_normalizer.distance_normalizer.normalize_scale(
                q_pred
            )
            q_target = self.tdm_normalizer.distance_normalizer.normalize_scale(
                q_target
            )
        bellman_errors = (q_pred - q_target) ** 2
        qf_loss = self.qf_criterion(q_pred, q_target)

        """
        Update Networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self._update_target_networks()

        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['Raw Policy Loss'] = np.mean(ptu.get_numpy(
                raw_policy_loss
            ))
            self.eval_statistics['Preactivation Policy Loss'] = (
                    self.eval_statistics['Policy Loss'] -
                    self.eval_statistics['Raw Policy Loss']
            )
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors',
                ptu.get_numpy(bellman_errors),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))

    def get_batch(self):
        batch = self.replay_buffer.random_batch(self.batch_size)

        """
        Update the goal states/rewards
        """
        num_steps_left = np.random.randint(
            0, self.max_tau + 1, (self.batch_size, 1)
        )
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = batch['resampled_goals']
        rewards = self._compute_scaled_rewards_np(
            batch, obs, actions, next_obs, goals
        )
        terminals = batch['terminals']

        terminals = 1 - (1 - terminals) * (num_steps_left != 0)
        rewards = rewards * terminals

        """
        Update the batch
        """
        batch['rewards'] = rewards
        batch['terminals'] = terminals
        batch['actions'] = actions
        batch['num_steps_left'] = num_steps_left
        batch['goals'] = goals
        batch['observations'] = obs
        batch['next_observations'] = next_obs

        return np_to_pytorch_batch(batch)

    def _compute_scaled_rewards_np(self, batch, obs, actions, next_obs, goals):
        """
        Rewards should be already multiplied by the reward scale and/or other
        factors. In other words, the rewards returned here should be
        immediately ready for any down-stream learner to consume.
        """
        neg_distances = self._compute_unscaled_neg_distances(next_obs, goals)
        return neg_distances * self.reward_scale

    def _compute_unscaled_neg_distances(self, next_obs, goals):
        diff = self.env.convert_obs_to_goals(next_obs) - goals
        if self.goal_weights is not None:
            diff = diff * self.goal_weights
        else:
            diff = diff * self.env.goal_dim_weights
        if self.vectorized:
            raw_neg_distances = -np.abs(diff)
        else:
            raw_neg_distances = -np.linalg.norm(
                diff,
                ord=1,
                axis=1,
                keepdims=True,
            )
        return raw_neg_distances

    def _sample_goal_for_rollout(self):
        return self.env.sample_goal_for_rollout()

    def _sample_max_tau_for_rollout(self):
        return self.max_tau

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        self._current_path_goal = self._sample_goal_for_rollout()
        self.training_env.set_goal(self._current_path_goal)
        self._rollout_tau = np.array([self.max_tau])
        return self.training_env.reset()

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
            num_steps_left=self._rollout_tau,
            goals=self._current_path_goal,
        )
        self._rollout_tau -= 1
        if self._rollout_tau[0] < 0:
            self._rollout_tau = np.array([self.max_tau])

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        return self.exploration_policy.get_action(
            observation,
            self._current_path_goal,
            self._rollout_tau,
        )

    def _handle_rollout_ending(self):
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self.replay_buffer.add_path(path)
            self._exploration_paths.append(path)
            self._current_path_builder = PathBuilder()

    def pretrain(self):
        if self.num_pretrain_paths == 0:
            return

        paths = []
        random_policy = RandomUniversalPolicy(self.env.action_space)
        for _ in range(self.num_pretrain_paths):
            goal = self.env.sample_goal_for_rollout()
            path = multitask_rollout(
                self.training_env,
                random_policy,
                goal=goal,
                init_tau=0,
                max_path_length=self.max_path_length,
            )
            paths.append(path)

        obs = np.vstack([path["observations"] for path in paths])
        self.pretrain_obs = obs
        if self.num_pretrain_paths == 0:
            return
        next_obs = np.vstack([path["next_observations"] for path in paths])
        actions = np.vstack([path["actions"] for path in paths])
        goals = np.vstack([path["goals"] for path in paths])
        neg_distances = self._compute_unscaled_neg_distances(next_obs, goals)

        ob_mean = np.mean(obs, axis=0)
        ob_std = np.std(obs, axis=0)
        ac_mean = np.mean(actions, axis=0)
        ac_std = np.std(actions, axis=0)
        new_goals = np.vstack([
            self._sample_goal_for_rollout()
            for _ in range(
                self.num_pretrain_paths * self.max_path_length
            )
        ])
        goal_mean = np.mean(new_goals, axis=0)
        goal_std = np.std(new_goals, axis=0)
        distance_mean = np.mean(neg_distances, axis=0)
        distance_std = np.std(neg_distances, axis=0)

        if self.tdm_normalizer is not None:
            self.tdm_normalizer.obs_normalizer.set_mean(ob_mean)
            self.tdm_normalizer.obs_normalizer.set_std(ob_std)
            self.tdm_normalizer.action_normalizer.set_mean(ac_mean)
            self.tdm_normalizer.action_normalizer.set_std(ac_std)
            self.tdm_normalizer.goal_normalizer.set_mean(goal_mean)
            self.tdm_normalizer.goal_normalizer.set_std(goal_std)
            if self.normalize_distance:
                self.tdm_normalizer.distance_normalizer.set_mean(distance_mean)
                self.tdm_normalizer.distance_normalizer.set_std(distance_std)

        if self.qf.tdm_normalizer is not None:
            self.target_qf.tdm_normalizer.copy_stats(
                self.qf.tdm_normalizer
            )
            self.target_policy.tdm_normalizer.copy_stats(
                self.qf.tdm_normalizer
            )
