from collections import deque, OrderedDict

import numpy as np

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.data_collector.base import StepCollector


class MdpStepCollector(StepCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._obs = None  # cache variable

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._obs = None

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )

    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        for _ in range(num_steps):
            self.collect_one_step(max_path_length, discard_incomplete_paths)

    def collect_one_step(
            self,
            max_path_length,
            discard_incomplete_paths,
    ):
        if self._obs is None:
            self._start_new_rollout()

        action, agent_info = self._policy.get_action(self._obs)
        next_ob, reward, terminal, env_info = (
            self._env.step(action)
        )
        if self._render:
            self._env.render(**self._render_kwargs)
        terminal = np.array([terminal])
        reward = np.array([reward])
        # store path obs
        self._current_path_builder.add_all(
            observations=self._obs,
            actions=action,
            rewards=reward,
            next_observations=next_ob,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        if terminal or len(self._current_path_builder) >= max_path_length:
            self._handle_rollout_ending(max_path_length,
                                        discard_incomplete_paths)
            self._start_new_rollout()
        else:
            self._obs = next_ob

    def _start_new_rollout(self):
        self._current_path_builder = PathBuilder()
        self._obs = self._env.reset()

    def _handle_rollout_ending(
            self,
            max_path_length,
            discard_incomplete_paths
    ):
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                return
            self._epoch_paths.append(path)
            self._num_paths_total += 1
            self._num_steps_total += path_len


class GoalConditionedStepCollector(StepCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._obs = None  # cache variable

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._obs = None

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )

    def start_collection(self):
        self._start_new_rollout()

    def end_collection(self):
        epoch_paths = self.get_epoch_paths()
        return epoch_paths

    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        for _ in range(num_steps):
            self.collect_one_step(max_path_length, discard_incomplete_paths)

    def collect_one_step(
            self,
            max_path_length,
            discard_incomplete_paths,
    ):
        if self._obs is None:
            self._start_new_rollout()

        new_obs = np.hstack((
            self._obs[self._observation_key],
            self._obs[self._desired_goal_key],
        ))
        action, agent_info = self._policy.get_action(new_obs)
        next_ob, reward, terminal, env_info = (
            self._env.step(action)
        )
        if self._render:
            self._env.render(**self._render_kwargs)
        terminal = np.array([terminal])
        reward = np.array([reward])
        # store path obs
        self._current_path_builder.add_all(
            observations=self._obs,
            actions=action,
            rewards=reward,
            next_observations=next_ob,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        if terminal or len(self._current_path_builder) >= max_path_length:
            self._handle_rollout_ending(max_path_length,
                                        discard_incomplete_paths)
            self._start_new_rollout()
        else:
            self._obs = next_ob

    def _start_new_rollout(self):
        self._current_path_builder = PathBuilder()
        self._obs = self._env.reset()

    def _handle_rollout_ending(
            self,
            max_path_length,
            discard_incomplete_paths
    ):
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                return
            self._epoch_paths.append(path)
            self._num_paths_total += 1
            self._num_steps_total += path_len


class ObsDictStepCollector(StepCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            observation_key='observation',
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._observation_key = observation_key

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._obs = None  # cache variable

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._obs = None

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
        )

    def start_collection(self):
        self._start_new_rollout()

    def end_collection(self):
        epoch_paths = self.get_epoch_paths()
        return epoch_paths

    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        for _ in range(num_steps):
            self.collect_one_step(max_path_length, discard_incomplete_paths)

    def collect_one_step(
            self,
            max_path_length,
            discard_incomplete_paths,
    ):
        if self._obs is None:
            self._start_new_rollout()

        new_obs = self._obs[self._observation_key]
        action, agent_info = self._policy.get_action(new_obs)
        next_ob, reward, terminal, env_info = (
            self._env.step(action)
        )
        if self._render:
            self._env.render(**self._render_kwargs)
        terminal = np.array([terminal])
        reward = np.array([reward])
        # store path obs
        self._current_path_builder.add_all(
            observations=self._obs,
            actions=action,
            rewards=reward,
            next_observations=next_ob,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        if terminal or len(self._current_path_builder) >= max_path_length:
            self._handle_rollout_ending(max_path_length,
                                        discard_incomplete_paths)
            self._start_new_rollout()
        else:
            self._obs = next_ob

    def _start_new_rollout(self):
        self._current_path_builder = PathBuilder()
        self._obs = self._env.reset()

    def _handle_rollout_ending(
            self,
            max_path_length,
            discard_incomplete_paths
    ):
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                return
            self._epoch_paths.append(path)
            self._num_paths_total += 1
            self._num_steps_total += path_len

