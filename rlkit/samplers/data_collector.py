import abc
from collections import deque, OrderedDict

from rlkit.samplers.rollout_functions import rollout, multitask_rollout


class PathCollector(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def collect_new_paths(self, max_path_length, num_steps):
        pass

    @abc.abstractmethod
    def get_epoch_paths(self):
        pass

    def end_epoch(self, epoch):
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}


class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=32,
    ):
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(self, max_path_length, num_steps):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            path = rollout(
                self._env,
                self._policy,
                max_path_length=min(  # Do not go over num_steps
                    max_path_length,
                    num_steps - num_steps_collected,
                ),
            )
            num_steps_collected += len(path['actions'])
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        return OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )


class GoalConditionedPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=32,
            observation_key='observation',
            desired_goal_key='desired_goal',
    ):
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(self, max_path_length, num_steps):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            path = multitask_rollout(
                self._env,
                self._policy,
                max_path_length=min(  # Do not go over num_steps
                    max_path_length,
                    num_steps - num_steps_collected,
                ),
                observation_key=self._observation_key,
                desired_goal_key=self._desired_goal_key,
                return_dict_obs=True,
            )
            num_steps_collected += len(path['actions'])
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        return OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )

