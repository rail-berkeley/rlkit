from collections import OrderedDict

import numpy as np
from gym.envs.mujoco import HalfCheetahEnv

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.serializable import Serializable
from rlkit.core import logger as default_logger
from rlkit.samplers.util import get_stat_in_paths
from rlkit.torch.tdm.envs.multitask_env import MultitaskEnv

MAX_SPEED = 6


class GoalXVelHalfCheetah(HalfCheetahEnv, MultitaskEnv, Serializable):
    def __init__(self):
        Serializable.quick_init(self, locals())
        self.target_x_vel = np.random.uniform(-MAX_SPEED, MAX_SPEED)
        super().__init__()
        MultitaskEnv.__init__(self)
        self.set_goal(np.array([5]))

    @property
    def goal_dim(self) -> int:
        return 1

    def sample_goals(self, batch_size):
        return np.random.uniform(-MAX_SPEED, MAX_SPEED, (batch_size, 1))

    def convert_obs_to_goals(self, obs):
        return obs[:, 8:9]

    def set_goal(self, goal):
        MultitaskEnv.set_goal(self, goal)
        self.target_x_vel = goal

    def step(self, action):
        ob, _, done, info_dict = super().step(action)
        xvel = ob[8]
        desired_xvel = self.target_x_vel
        xvel_error = np.linalg.norm(xvel - desired_xvel)
        reward = - xvel_error
        info_dict['xvel'] = xvel
        info_dict['desired_xvel'] = desired_xvel
        info_dict['xvel_error'] = xvel_error
        return ob, reward, done, info_dict

    def log_diagnostics(self, paths, logger=default_logger):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)
        xvels = get_stat_in_paths(
            paths, 'env_infos', 'xvel'
        )
        desired_xvels = get_stat_in_paths(
            paths, 'env_infos', 'desired_xvel'
        )
        xvel_errors = get_stat_in_paths(
            paths, 'env_infos', 'xvel_error'
        )

        statistics = OrderedDict()
        for stat, name in [
            (xvels, 'xvels'),
            (desired_xvels, 'desired xvels'),
            (xvel_errors, 'xvel errors'),
        ]:
            statistics.update(create_stats_ordered_dict(
                '{}'.format(name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final {}'.format(name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)
