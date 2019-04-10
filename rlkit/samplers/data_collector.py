import abc
import numpy as np
from collections import deque, OrderedDict

from rlkit.envs.vae_wrapper import VAEWrappedEnv
from rlkit.data_management.path_builder import PathBuilder
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
            online_mode=False,
    ):
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

        self._num_steps_total = 0
        self._num_paths_total = 0
        self.online_mode = online_mode

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
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

    def set_online(self, value=True):
        self.online_mode = value

    def start_collection(self):
        assert self.online_mode, 'Step-wise collection is only enabled in online mode.'
        self._current_path_builder = PathBuilder()
        self._paths = []
        self._obs = self._start_new_rollout()

    def end_collection(self):
        assert self.online_mode, 'Step-wise collection is only enabled in online mode.'
        paths = self._paths
        self._num_paths_total += len(paths)
        self._num_steps_total += sum([ len(path['actions'] for path in paths]))
        self._epoch_paths.extend(paths)
        # clear paths
        self._paths = []
        return paths

    def collect_new_step(self, max_path_length, num_steps):
        assert self.online_mode, 'Step-wise collection is only enabled in online mode.'
        # rollout step
        action, agent_info = self._policy.get_action(self._obs)
        next_ob, reward, terminal, env_info = (
            self._env.step(action)
        )
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
            self._handle_rollout_ending()
            self._obs = self._start_new_rollout()
        else:
            self._obs = next_ob

    def _start_new_rollout(self):
        return self._env.reset()

    def _handle_rollout_ending(self):
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self._current_path_builder = PathBuilder()
            self._paths.append(path)



class GoalConditionedPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=32,
            observation_key='observation',
            desired_goal_key='desired_goal',
            online_mode=False,
    ):
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key

        self._num_steps_total = 0
        self._num_paths_total = 0
        self.set_online(online_mode)
        self._obs = None # cache variable

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        assert self.online_mode  == False
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
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
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

    # Functions related to online mode
    def set_online(self, value):
        self.online_mode = value

    def start_collection(self):
        assert self.online_mode, 'Step-wise collection is only enabled in online mode.'
        self._current_path_builder = PathBuilder()
        self._paths = []
        self._obs = self._start_new_rollout()

    def end_collection(self):
        assert self.online_mode, 'Step-wise collection is only enabled in online mode.'
        paths = self._paths
        self._num_paths_total += len(paths)
        self._num_steps_total += sum([ len(path['actions'] for path in paths]))
        self._epoch_paths.extend(paths)
        # clear paths
        self._paths = []
        return paths

    def collect_new_step(self, max_path_length, num_steps):
        assert self.online_mode, 'Step-wise collection is only enabled in online mode.'
        # rollout step
        new_obs = np.hstack((
            self._obs[self._observation_key],
            self._obs[self._desired_goal_key],
        ))
        action, agent_info = self._policy.get_action(new_obs)
        next_ob, reward, terminal, env_info = (
            self._env.step(action)
        )
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
            self._handle_rollout_ending()
            self._obs = self._start_new_rollout()
        else:
            self._obs = next_ob

    def _start_new_rollout(self):
        return self._env.reset()

    def _handle_rollout_ending(self):
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self._current_path_builder = PathBuilder()
            self._paths.append(path)

class VAEWrappedEnvPathCollector(GoalConditionedPathCollector):
    def __init__(
            self,
            goal_sampling_mode,
            env: VAEWrappedEnv,
            policy,
            decode_goals=False,
            **kwargs
    ):
        super().__init__(env, policy, **kwargs)
        self._goal_sampling_mode = goal_sampling_mode
        self._decode_goals = decode_goals

<<<<<<< HEAD
    def collect_new_paths(self, *args, **kwargs):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        self._env.decode_goals = self._decode_goals
        return super().collect_new_paths(*args, **kwargs)
=======
    def collect_new_paths(self, max_path_length, num_steps):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        self._env.decode_goals = self._decode_goals
        return super().collect_new_paths(max_path_length, num_steps)
>>>>>>> Add online rl algorithm + train_mode function
