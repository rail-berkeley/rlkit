from collections import OrderedDict

import numpy as np
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box

from rlkit.core import logger as default_logger
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.serializable import Serializable
from rlkit.envs.mujoco_env import get_asset_xml
from rlkit.samplers.util import get_stat_in_paths
from rlkit.torch.tdm.envs.multitask_env import MultitaskEnv


class Reacher7DofMultitaskEnv(
    MultitaskEnv, mujoco_env.MujocoEnv, Serializable
):
    def __init__(self, distance_metric_order=None, goal_dim_weights=None):
        self._desired_xyz = np.zeros(3)
        Serializable.quick_init(self, locals())
        MultitaskEnv.__init__(
            self,
            distance_metric_order=distance_metric_order,
            goal_dim_weights=goal_dim_weights,
        )
        mujoco_env.MujocoEnv.__init__(
            self,
            get_asset_xml('reacher_7dof.xml'),
            5,
        )
        self.observation_space = Box(
            np.array([
                -2.28, -0.52, -1.4, -2.32, -1.5, -1.094, -1.5,  # joint
                -3, -3, -3, -3, -3, -3, -3, # velocity
                -0.75, -1.25, -0.2,  # EE xyz

            ]),
            np.array([
                1.71, 1.39, 1.7, 0, 1.5, 0, 1.5,  # joints
                3, 3, 3, 3, 3, 3, 3,  # velocity
                0.75, 0.25, 0.6,  # EE xyz
            ])
        )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005, size=self.model.nv)
        qvel[-7:] = 0
        self.set_state(qpos, qvel)
        self._set_goal_xyz(self._desired_xyz)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
        ])

    def _step(self, a):
        distance = np.linalg.norm(
            self.get_body_com("tips_arm") - self._desired_xyz
        )
        reward = - distance
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(
            distance=distance,
            multitask_goal=self.multitask_goal,
            desired_xyz=self._desired_xyz,
            goal=self.multitask_goal,
        )

    def _set_goal_xyz(self, xyz_pos):
        current_qpos = self.model.data.qpos.flat
        current_qvel = self.model.data.qvel.flat.copy()
        new_qpos = current_qpos.copy()
        new_qpos[-7:-4] = xyz_pos
        self._desired_xyz = xyz_pos
        self.set_state(new_qpos, current_qvel)

    def log_diagnostics(self, paths, logger=default_logger):
        super().log_diagnostics(paths)
        statistics = OrderedDict()

        euclidean_distances = get_stat_in_paths(
            paths, 'env_infos', 'distance'
        )
        statistics.update(create_stats_ordered_dict(
            'Euclidean distance to goal', euclidean_distances
        ))
        statistics.update(create_stats_ordered_dict(
            'Final Euclidean distance to goal',
            [d[-1] for d in euclidean_distances],
            always_show_all_stats=True,
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    def joints_to_full_state(self, joints):
        current_qpos = self.model.data.qpos.flat.copy()
        current_qvel = self.model.data.qvel.flat.copy()

        new_qpos = current_qpos.copy()
        new_qpos[:7] = joints
        self.set_state(new_qpos, current_qvel)
        full_state = self._get_obs().copy()
        self.set_state(current_qpos, current_qvel)
        return full_state


class Reacher7DofFullGoal(Reacher7DofMultitaskEnv):
    @property
    def goal_dim(self) -> int:
        return 17

    def sample_goals(self, batch_size):
        return self.sample_states(batch_size)

    def convert_obs_to_goals(self, obs):
        return obs

    def set_goal(self, goal):
        super().set_goal(goal)
        self._set_goal_xyz_automatically(goal)

    def modify_goal_for_rollout(self, goal):
        goal[7:14] = 0
        return goal

    def _set_goal_xyz_automatically(self, goal):
        current_qpos = self.model.data.qpos.flat.copy()
        current_qvel = self.model.data.qvel.flat.copy()

        new_qpos = current_qpos.copy()
        new_qpos[:7] = goal[:7]
        self.set_state(new_qpos, current_qvel)
        goal_xyz = self.get_body_com("tips_arm").copy()
        self.set_state(current_qpos, current_qvel)

        self._set_goal_xyz(goal_xyz)
        self.multitask_goal[14:17] = goal_xyz

    def sample_states(self, batch_size):
        random_pos = np.random.uniform(
            [-2.28, -0.52, -1.4, -2.32, -1.5, -1.094, -1.5],
            [1.71, 1.39, 1.7, 0, 1.5, 0, 1.5, ],
            (batch_size, 7)
        )
        random_vel = np.random.uniform(-3, 3, (batch_size, 7))
        random_xyz = np.random.uniform(
            np.array([-0.75, -1.25, -0.2]),
            np.array([0.75, 0.25, 0.6]),
            (batch_size, 3)
        )
        return np.hstack((
            random_pos,
            random_vel,
            random_xyz,
        ))

    def cost_fn(self, states, actions, next_states):
        """
        This is added for model-based code. This is COST not reward.
        So lower is better.

        :param states:  (BATCH_SIZE x state_dim) numpy array
        :param actions:  (BATCH_SIZE x action_dim) numpy array
        :param next_states:  (BATCH_SIZE x state_dim) numpy array
        :return: (BATCH_SIZE, ) numpy array
        """
        if len(next_states.shape) == 1:
            next_states = np.expand_dims(next_states, 0)
        # xyz_pos = next_states[:, 14:17]
        # desired_xyz_pos = self.multitask_goal[14:17] * np.ones_like(xyz_pos)
        # diff = xyz_pos - desired_xyz_pos
        next_joint_angles = next_states[:, :7]
        desired_joint_angles = (
            self.multitask_goal[:7] * np.ones_like(next_joint_angles)
        )
        diff = next_joint_angles - desired_joint_angles
        return (diff**2).sum(1, keepdims=True)
