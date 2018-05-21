from collections import OrderedDict

import numpy as np

from rlkit.envs.ant import AntEnv

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.serializable import Serializable
from rlkit.core import logger as default_logger
from rlkit.samplers.util import get_stat_in_paths
from rlkit.torch.tdm.envs.multitask_env import MultitaskEnv


class GoalXYPosAnt(AntEnv, MultitaskEnv, Serializable):
    def __init__(self, min_distance=0, max_distance=2, use_low_gear_ratio=True):
        Serializable.quick_init(self, locals())
        self.max_distance = max_distance
        self.min_distance = min_distance
        MultitaskEnv.__init__(self)
        super().__init__(use_low_gear_ratio=use_low_gear_ratio)
        self.set_goal(np.array([self.max_distance, self.max_distance]))

    @property
    def goal_dim(self) -> int:
        return 2

    def sample_goals(self, batch_size):
        raise NotImplementedError()

    def sample_goal_for_rollout(self):
        goal = np.random.uniform(-self.max_distance, self.max_distance, 2)
        while np.linalg.norm(goal) < self.min_distance:
            goal = np.random.uniform(-self.max_distance, self.max_distance, 2)
        return goal

    def set_goal(self, goal):
        super().set_goal(goal)
        site_pos = self.model.site_pos.copy()
        site_pos[0, 0:2] = goal
        site_pos[0, 2] = 0.5
        self.model.site_pos[:] = site_pos

    def convert_obs_to_goals(self, obs):
        return obs[:, 27:29]

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso"),
        ])

    def step(self, action):
        ob, _, done, info_dict = super().step(action)
        xy_pos = self.convert_ob_to_goal(ob)
        pos_error = np.linalg.norm(xy_pos - self.multitask_goal)
        reward = - pos_error
        info_dict['x_pos'] = xy_pos[0]
        info_dict['y_pos'] = xy_pos[1]
        info_dict['dist_from_origin'] = np.linalg.norm(xy_pos)
        info_dict['desired_x_pos'] = self.multitask_goal[0]
        info_dict['desired_y_pos'] = self.multitask_goal[1]
        info_dict['desired_dist_from_origin'] = (
            np.linalg.norm(self.multitask_goal)
        )
        info_dict['pos_error'] = pos_error
        info_dict['goal'] = self.multitask_goal
        return ob, reward, done, info_dict

    def sample_states(self, batch_size):
        raise NotImplementedError()

    def log_diagnostics(self, paths, logger=default_logger):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)

        statistics = OrderedDict()
        for name_in_env_infos, name_to_log in [
            ('x_pos', 'X Position'),
            ('y_pos', 'Y Position'),
            ('dist_from_origin', 'Distance from Origin'),
            ('desired_x_pos', 'Desired X Position'),
            ('desired_y_pos', 'Desired Y Position'),
            ('desired_dist_from_origin', 'Desired Distance from Origin'),
            ('pos_error', 'Distance to goal'),
        ]:
            stat = get_stat_in_paths(paths, 'env_infos', name_in_env_infos)
            statistics.update(create_stats_ordered_dict(
                name_to_log,
                stat,
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
        for name_in_env_infos, name_to_log in [
            ('dist_from_origin', 'Distance from Origin'),
            ('desired_dist_from_origin', 'Desired Distance from Origin'),
            ('pos_error', 'Distance to goal'),
        ]:
            stat = get_stat_in_paths(paths, 'env_infos', name_in_env_infos)
            statistics.update(create_stats_ordered_dict(
                'Final {}'.format(name_to_log),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def __getstate__(self):
        return Serializable.__getstate__(self)

    def __setstate__(self, state):
        return Serializable.__setstate__(self, state)


class GoalXYPosAndVelAnt(AntEnv, MultitaskEnv, Serializable):
    def __init__(
            self,
            max_speed=0.05,
            max_distance=1,
            use_low_gear_ratio=True,
            speed_weight=0.9,
            done_threshold=0.005,
            goal_dim_weights=None,
    ):
        Serializable.quick_init(self, locals())
        self.max_distance = max_distance
        self.max_speed = max_speed
        self.speed_weight = speed_weight
        self.done_threshold = done_threshold
        self.initializing = True
        # TODO: fix this hack
        if speed_weight is None:
            self.speed_weight = 0.9  # just for init to work
        MultitaskEnv.__init__(self, goal_dim_weights=goal_dim_weights)
        super().__init__(use_low_gear_ratio=use_low_gear_ratio)
        self.set_goal(np.array([
            self.max_distance,
            self.max_distance,
            self.max_speed,
            self.max_speed,
        ]))
        self.initializing = False
        if speed_weight is None:
            assert (
                           self.goal_dim_weights[0] == self.goal_dim_weights[1]
                   ) and (
                           self.goal_dim_weights[2] == self.goal_dim_weights[3]
                   )
            self.speed_weight = self.goal_dim_weights[2]
        assert 0 <= self.speed_weight <= 1

    @property
    def goal_dim(self) -> int:
        return 4

    def sample_goals(self, batch_size):
        return np.random.uniform(
            np.array([
                -self.max_distance,
                -self.max_distance,
                -self.max_speed,
                -self.max_speed
            ]),
            np.array([
                self.max_distance,
                self.max_distance,
                self.max_speed,
                self.max_speed
            ]),
            (batch_size, 4),
        )

    def convert_obs_to_goals(self, obs):
        return np.hstack((
            obs[:, 27:29],
            obs[:, 30:32],
        ))

    def set_goal(self, goal):
        super().set_goal(goal)
        site_pos = self.model.site_pos.copy()
        site_pos[0, 0:2] = goal[:2]
        site_pos[0, 2] = 0.5
        self.model.site_pos[:] = site_pos

    def _get_obs(self):
        raise NotImplementedError()

    def step(self, action):
        # get_body_comvel doesn't work, so you need to save the last position
        torso_xyz_before = self.get_body_com("torso")
        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = self.get_body_com("torso")
        torso_velocity = torso_xyz_after - torso_xyz_before

        done = False

        ob = np.hstack((
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso"),
            torso_velocity,
        ))

        error = self.convert_ob_to_goal(ob) - self.multitask_goal
        pos_error = np.linalg.norm(error[:3])
        vel_error = np.linalg.norm(error[3:])
        weighted_vel_error = vel_error * self.speed_weight
        weighted_pos_error = pos_error * (1 - self.speed_weight)
        reward = - (weighted_pos_error + weighted_vel_error)
        if np.abs(reward) < self.done_threshold and not self.initializing:
            done = True
        info_dict = dict(
            goal=self.multitask_goal,
            vel_error=vel_error,
            pos_error=pos_error,
            weighted_vel_error=weighted_vel_error,
            weighted_pos_error=weighted_pos_error,
        )
        return ob, reward, done, info_dict

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return np.hstack((
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso"),
            np.zeros(3),  # init velocity is zero
        ))

    def sample_states(self, batch_size):
        raise NotImplementedError()

    def log_diagnostics(self, paths, logger=default_logger):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)

        statistics = OrderedDict()
        for stat_name in [
            'pos_error',
            'vel_error',
            'weighted_pos_error',
            'weighted_vel_error',
        ]:
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '{}'.format(stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final {}'.format(stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        weighted_error = (
            get_stat_in_paths(paths, 'env_infos', 'weighted_pos_error')
            + get_stat_in_paths(paths, 'env_infos', 'weighted_vel_error')
        )
        statistics.update(create_stats_ordered_dict(
            "Weighted Error",
            weighted_error,
            always_show_all_stats=True,
        ))
        statistics.update(create_stats_ordered_dict(
            "Final Weighted Error",
            [s[-1] for s in weighted_error],
            always_show_all_stats=True,
        ))

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def __getstate__(self):
        return Serializable.__getstate__(self)

    def __setstate__(self, state):
        return Serializable.__setstate__(self, state)


