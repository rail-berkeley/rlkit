import abc
import io
import warnings
import xml.etree.ElementTree as ET
from os import path

import cv2
import gym
import mujoco_py
import numpy as np
import quaternion
import robosuite
from d4rl.kitchen.adept_envs.simulation import module
from d4rl.kitchen.adept_envs.simulation.renderer import DMRenderer, MjPyRenderer
from gym.spaces.box import Box
from metaworld.envs.mujoco.mujoco_env import MujocoEnv, _assert_task_is_set
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerMocapBase,
    SawyerXYZEnv,
)
from robosuite.utils import macros
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
from robosuite.utils.mujoco_py_renderer import MujocoPyRenderer
from robosuite.utils.observables import Observable, sensor


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        gym.Wrapper.__init__(self, env)
        self._duration = duration
        self._max_episode_steps = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, amount):
        gym.Wrapper.__init__(self, env)
        self._amount = amount

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        o, r, d, i = self.env.step(original)
        return o, r, d, i

    def reset(self):
        return self.env.reset()


class ImageUnFlattenWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env.max_steps
        self.observation_space = Box(
            0, 255, (3, self.env.imwidth, self.env.imheight), dtype=np.uint8
        )
        self.reward_ctr = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs = self.env.reset()
        return obs.reshape(-1, self.env.imwidth, self.env.imheight)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return (
            obs.reshape(-1, self.env.imwidth, self.env.imheight),
            reward,
            done,
            info,
        )


class ImageTransposeWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = Box(
            0, 255, (self.env.imwidth, self.env.imheight, 3), dtype=np.uint8
        )
        self.reward_ctr = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs = self.env.reset()
        return obs.reshape(-1, self.env.imwidth, self.env.imheight).transpose(1, 2, 0)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return (
            obs.reshape(-1, self.env.imwidth, self.env.imheight).transpose(1, 2, 0),
            reward,
            done,
            info,
        )


class MetaworldWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._elapsed_steps = 0

    def reset(self):
        obs = super().reset()
        self._elapsed_steps = 0
        return obs

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="human",
        render_im_shape=(1000, 1000),
    ):
        o, r, d, i = self.env.step(
            action,
            # render_every_step=render_every_step,
            # render_mode=render_mode,
            # render_im_shape=render_im_shape,
        )
        new_i = {}
        for k, v in i.items():
            if v is not None:
                new_i[k] = v
        self._elapsed_steps += 1
        return o, r, d, new_i


class ImageEnvMetaworld(gym.Wrapper):
    def __init__(
        self,
        env,
        imwidth=84,
        imheight=84,
        reward_scale=1.0,
    ):
        gym.Wrapper.__init__(self, env)
        self.max_steps = self.env.max_path_length
        self.imwidth = imwidth
        self.imheight = imheight
        self.observation_space = Box(
            0, 255, (3 * self.imwidth * self.imheight,), dtype=np.uint8
        )
        self.image_shape = (3, self.imwidth, self.imheight)
        self.num_steps = 0
        self.reward_scale = reward_scale

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _get_image(self):
        # use this if using dm control backend!
        if hasattr(self.env, "_use_dm_backend"):
            img = self.env.render(
                mode="rgb_array", width=self.imwidth, height=self.imheight
            )
        else:
            img = self.env.sim.render(
                width=self.imwidth,
                height=self.imheight,
            )

        img = img.transpose(2, 0, 1).flatten()
        return img

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="human",
        render_im_shape=(1000, 1000),
    ):
        o, r, d, i = self.env.step(
            action,
            # render_every_step=render_every_step,
            # render_mode=render_mode,
            # render_im_shape=render_im_shape,
        )
        self.num_steps += 1
        o = self._get_image()
        r = self.reward_scale * r
        new_i = {}
        for k, v in i.items():
            if v is not None:
                new_i[k] = v
        return o, r, d, new_i

    def reset(self):
        super().reset()
        self.num_steps = 0
        return self._get_image()


class DictObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = {}
        spaces["image"] = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(
        self,
        action,
    ):
        o, r, d, i = super().step(action)
        return {"image": o}, r, d, i

    def reset(self):
        return {"image": self.env.reset()}


class SawyerXYZEnvMetaworldPrimitives(SawyerXYZEnv):
    def reset_action_space(
        self,
        control_mode="end_effector",
        use_combined_action_space=True,
        action_scale=1 / 100,
        max_path_length=500,
        remove_rotation_primitives=True,
    ):
        self.max_path_length = max_path_length
        self.action_scale = action_scale

        # primitives
        if remove_rotation_primitives:
            self.primitive_idx_to_name = {
                0: "move_delta_ee_pose",
                1: "lift",
                2: "drop",
                3: "move_left",
                4: "move_right",
                5: "move_forward",
                6: "move_backward",
                7: "open_gripper",
                8: "close_gripper",
            }
            self.primitive_name_to_func = dict(
                move_delta_ee_pose=self.move_delta_ee_pose,
                lift=self.lift,
                drop=self.drop,
                move_left=self.move_left,
                move_right=self.move_right,
                move_forward=self.move_forward,
                move_backward=self.move_backward,
                open_gripper=self.open_gripper,
                close_gripper=self.close_gripper,
            )
            self.primitive_name_to_action_idx = dict(
                move_delta_ee_pose=[0, 1, 2],
                lift=3,
                drop=4,
                move_left=5,
                move_right=6,
                move_forward=7,
                move_backward=8,
                open_gripper=[],  # doesn't matter
                close_gripper=[],  # doesn't matter
            )
            self.max_arg_len = 9
        else:
            self.primitive_idx_to_name = {
                0: "angled_x_y_grasp",
                1: "move_delta_ee_pose",
                2: "rotate_about_y_axis",
                3: "lift",
                4: "drop",
                5: "move_left",
                6: "move_right",
                7: "move_forward",
                8: "move_backward",
                9: "open_gripper",
                10: "close_gripper",
                11: "rotate_about_x_axis",
            }
            self.primitive_name_to_func = dict(
                angled_x_y_grasp=self.angled_x_y_grasp,
                move_delta_ee_pose=self.move_delta_ee_pose,
                rotate_about_y_axis=self.rotate_about_y_axis,
                lift=self.lift,
                drop=self.drop,
                move_left=self.move_left,
                move_right=self.move_right,
                move_forward=self.move_forward,
                move_backward=self.move_backward,
                open_gripper=self.open_gripper,
                close_gripper=self.close_gripper,
                rotate_about_x_axis=self.rotate_about_x_axis,
            )
            self.primitive_name_to_action_idx = dict(
                angled_x_y_grasp=[0, 1, 2],
                move_delta_ee_pose=[3, 4, 5],
                rotate_about_y_axis=6,
                lift=7,
                drop=8,
                move_left=9,
                move_right=10,
                move_forward=11,
                move_backward=12,
                rotate_about_x_axis=13,
                open_gripper=[],  # doesn't matter
                close_gripper=[],  # doesn't matter
            )
            self.max_arg_len = 14
        self.num_primitives = len(self.primitive_name_to_func)
        self.control_mode = control_mode

        combined_action_space_low = -1 * np.ones(self.max_arg_len)
        combined_action_space_high = np.ones(self.max_arg_len)
        self.combined_action_space = Box(
            combined_action_space_low, combined_action_space_high, dtype=np.float32
        )
        self.use_combined_action_space = use_combined_action_space
        self.fixed_schema = False
        if self.use_combined_action_space and self.control_mode == "primitives":
            self.action_space = self.combined_action_space
            act_lower_primitive = np.zeros(self.num_primitives)
            act_upper_primitive = np.ones(self.num_primitives)
            act_lower = np.concatenate((act_lower_primitive, self.action_space.low))
            act_upper = np.concatenate(
                (
                    act_upper_primitive,
                    self.action_space.high,
                )
            )
            self.action_space = Box(act_lower, act_upper, dtype=np.float32)

    @_assert_task_is_set
    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        a = np.clip(action, -1.0, 1.0)
        if self.control_mode in [
            "joint_position",
            "joint_velocity",
            "torque",
            "end_effector",
        ]:
            if self.control_mode == "end_effector":
                self.set_xyz_action(a[:3])
                self.do_simulation([a[-1], -a[-1]])
        else:
            self.img_array = []
            self.act(
                a,
                render_every_step=render_every_step,
                render_mode=render_mode,
                render_im_shape=render_im_shape,
            )

        self.curr_path_length += 1

        # Running the simulator can sometimes mess up site positions, so
        # re-position them here to make sure they're accurate
        for site in self._target_site_config:
            self._set_pos_site(*site)

        if self._did_see_sim_exception:
            return (
                self._last_stable_obs,  # observation just before going unstable
                0.0,  # reward (penalize for causing instability)
                False,  # termination flag always False
                {  # info
                    "success": False,
                    "near_object": 0.0,
                    "grasp_success": False,
                    "grasp_reward": 0.0,
                    "in_place_reward": 0.0,
                    "obj_to_target": 0.0,
                    "unscaled_reward": 0.0,
                },
            )

        self._last_stable_obs = self._get_obs()
        if not self.isV2:
            # v1 environments expect this superclass step() to return only the
            # most recent observation. they override the rest of the
            # functionality and end up returning the same sort of tuple that
            # this does
            return self._last_stable_obs

        reward, info = self.evaluate_state(self._last_stable_obs, action)
        return self._last_stable_obs, reward, False, info

    def _get_site_pos(self, siteName):
        return self.data.site_xpos[self.model.site_name2id(siteName)]

    # primitives:
    def get_idx_from_primitive_name(self, primitive_name):
        for idx, pn in self.primitive_idx_to_name.items():
            if pn == primitive_name:
                return idx

    def get_site_xpos(self, name):
        id = self.sim.model.site_name2id(name)
        return self.sim.data.site_xpos[id]

    def get_mocap_pos(self, name):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        return self.sim.data.mocap_pos[mocap_id]

    def set_mocap_pos(self, name, value):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        self.sim.data.mocap_pos[mocap_id] = value

    def get_mocap_quat(self, name):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        return self.sim.data.mocap_quat[mocap_id]

    def set_mocap_quat(self, name, value):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        self.sim.data.mocap_quat[mocap_id] = value

    def ctrl_set_action(self, sim, action):
        self.sim.data.ctrl[0] = action[-2]
        self.sim.data.ctrl[1] = action[-1]

    def mocap_set_action(self, sim, action):
        if sim.model.nmocap > 0:
            action, _ = np.split(action, (sim.model.nmocap * 7,))
            action = action.reshape(sim.model.nmocap, 7)

            pos_delta = action[:, :3]
            quat_delta = action[:, 3:]
            new_mocap_pos = self.data.mocap_pos + pos_delta[None]
            new_mocap_quat = self.data.mocap_quat + quat_delta[None]

            new_mocap_pos[0, :] = np.clip(
                new_mocap_pos[0, :],
                self.mocap_low,
                self.mocap_high,
            )
            self.data.set_mocap_pos("mocap", new_mocap_pos)
            self.data.set_mocap_quat("mocap", new_mocap_quat)

    def reset_mocap2body_xpos(self, sim):
        if (
            sim.model.eq_type is None
            or sim.model.eq_obj1id is None
            or sim.model.eq_obj2id is None
        ):
            return
        for eq_type, obj1_id, obj2_id in zip(
            sim.model.eq_type, sim.model.eq_obj1id, sim.model.eq_obj2id
        ):
            if eq_type != mujoco_py.const.EQ_WELD:
                continue

            mocap_id = sim.model.body_mocapid[obj1_id]
            if mocap_id != -1:
                body_idx = obj2_id
            else:
                mocap_id = sim.model.body_mocapid[obj2_id]
                body_idx = obj1_id

            assert mocap_id != -1
            sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
            sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]

    def _set_action(self, action):
        assert action.shape == (9,)

        action = action.copy()
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:7], action[7:9]

        pos_ctrl *= 0.05
        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        self.ctrl_set_action(self.sim, action)
        self.mocap_set_action(self.sim, action)

    def rpy_to_quat(self, rpy):
        q = quaternion.from_euler_angles(rpy)
        return np.array([q.x, q.y, q.z, q.w])

    def quat_to_rpy(self, q):
        q = quaternion.quaternion(q[0], q[1], q[2], q[3])
        return quaternion.as_euler_angles(q)

    def convert_xyzw_to_wxyz(self, q):
        return np.array([q[3], q[0], q[1], q[2]])

    def close_gripper(
        self,
        unusued=None,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        for _ in range(200):
            self._set_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, -1]))
            self.data.set_mocap_quat("mocap", np.array([1, 0, 1, 0]))
            self.sim.step()
            if render_every_step:
                if render_mode == "rgb_array":
                    self.img_array.append(
                        self.render(
                            render_mode,
                            render_im_shape[0],
                            render_im_shape[1],
                        )
                    )
                else:
                    self.render(
                        render_mode,
                        render_im_shape[0],
                        render_im_shape[1],
                    )

    def open_gripper(
        self,
        unusued=None,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        for _ in range(200):
            self._set_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, 1]))
            self.data.set_mocap_quat("mocap", np.array([1, 0, 1, 0]))
            self.sim.step()
            if render_every_step:
                if render_mode == "rgb_array":
                    self.img_array.append(
                        self.render(
                            render_mode,
                            render_im_shape[0],
                            render_im_shape[1],
                        )
                    )
                else:
                    self.render(
                        render_mode,
                        render_im_shape[0],
                        render_im_shape[1],
                    )

    def rotate_ee(
        self,
        rpy,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        gripper = self.sim.data.qpos[7:9]
        for _ in range(200):
            quat = self.rpy_to_quat(rpy)
            quat_delta = self.convert_xyzw_to_wxyz(quat) - self.sim.data.body_xquat[10]
            self._set_action(
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        quat_delta[0],
                        quat_delta[1],
                        quat_delta[2],
                        quat_delta[3],
                        gripper[0],
                        gripper[1],
                    ]
                )
            )
            self.data.set_mocap_quat("mocap", np.array([1, 0, 1, 0]))
            self.sim.step()
            if render_every_step:
                if render_mode == "rgb_array":
                    self.img_array.append(
                        self.render(
                            render_mode,
                            render_im_shape[0],
                            render_im_shape[1],
                        )
                    )
                else:
                    self.render(
                        render_mode,
                        render_im_shape[0],
                        render_im_shape[1],
                    )

    def goto_pose(
        self,
        pose,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        # clamp the pose within workspace limits:
        gripper = self.sim.data.qpos[7:9]
        for _ in range(300):
            pose = np.clip(
                pose,
                self.mocap_low,
                self.mocap_high,
            )
            delta = pose - self.get_endeff_pos()
            self._set_action(
                np.array(
                    [
                        delta[0],
                        delta[1],
                        delta[2],
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        gripper[0],
                        gripper[1],
                    ]
                )
            )
            self.data.set_mocap_quat("mocap", np.array([1, 0, 1, 0]))
            self.sim.step()
            if render_every_step:
                if render_mode == "rgb_array":
                    self.img_array.append(
                        self.render(
                            render_mode,
                            render_im_shape[0],
                            render_im_shape[1],
                        )
                    )
                else:
                    self.render(
                        render_mode,
                        render_im_shape[0],
                        render_im_shape[1],
                    )

    def rotate_about_x_axis(
        self,
        angle,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        rotation = self.quat_to_rpy(self.sim.data.body_xquat[10]) - np.array(
            [angle, 0, 0]
        )
        self.rotate_ee(
            rotation,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )

    def angled_x_y_grasp(
        self,
        angle_and_xy,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        angle, x_dist, y_dist = angle_and_xy
        angle = np.clip(angle, -np.pi, np.pi)
        rotation = self.quat_to_rpy(self.sim.data.body_xquat[10]) - np.array(
            [angle, 0, 0]
        )
        self.rotate_ee(
            rotation,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        self.goto_pose(
            self.get_endeff_pos() + np.array([x_dist, 0.0, 0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, y_dist, 0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        self.close_gripper(
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )

    def move_delta_ee_pose(
        self,
        pose,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        self.goto_pose(
            self.get_endeff_pos() + pose,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )

    def rotate_about_y_axis(
        self,
        angle,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        angle = np.clip(angle, -np.pi, np.pi)
        rotation = self.quat_to_rpy(self.sim.data.body_xquat[10]) - np.array(
            [0, 0, angle],
        )
        self.rotate_ee(
            rotation,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )

    def lift(
        self,
        z_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        z_dist = np.maximum(z_dist, 0.0)
        self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, 0.0, z_dist]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )

    def drop(
        self,
        z_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        z_dist = np.maximum(z_dist, 0.0)
        self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, 0.0, -z_dist]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )

    def move_left(
        self,
        x_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        x_dist = np.maximum(x_dist, 0.0)
        self.goto_pose(
            self.get_endeff_pos() + np.array([-x_dist, 0.0, 0.0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )

    def move_right(
        self,
        x_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        x_dist = np.maximum(x_dist, 0.0)
        self.goto_pose(
            self.get_endeff_pos() + np.array([x_dist, 0.0, 0.0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )

    def move_forward(
        self,
        y_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        y_dist = np.maximum(y_dist, 0.0)
        self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, y_dist, 0.0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )

    def move_backward(
        self,
        y_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        y_dist = np.maximum(y_dist, 0.0)
        self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, -y_dist, 0.0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )

    def break_apart_action(self, a):
        broken_a = {}
        for k, v in self.primitive_name_to_action_idx.items():
            broken_a[k] = a[v]
        return broken_a

    def act(
        self,
        a,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        if not np.any(a):
            # all zeros should be a no-op!!!
            return
        a = np.clip(a, self.action_space.low, self.action_space.high)
        a = a * self.action_scale
        primitive_idx, primitive_args = (
            np.argmax(a[: self.num_primitives]),
            a[self.num_primitives :],
        )
        primitive_name = self.primitive_idx_to_name[primitive_idx]
        if primitive_name != "no_op":
            primitive_name_to_action_dict = self.break_apart_action(primitive_args)
            primitive_action = primitive_name_to_action_dict[primitive_name]
            primitive = self.primitive_name_to_func[primitive_name]
            primitive(
                primitive_action,
                render_every_step=render_every_step,
                render_mode=render_mode,
                render_im_shape=render_im_shape,
            )

    # (TODO): fix this for dm control backend
    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


class DMControlBackendMetaworldMujocoEnv(MujocoEnv):
    def __init__(self, model_path, frame_skip, rgb_array_res=(640, 480)):
        if not path.exists(model_path):
            raise IOError("File %s does not exist" % model_path)
        self.frame_skip = frame_skip
        self._use_dm_backend = True
        camera_settings = {}
        if self._use_dm_backend:
            dm_mujoco = module.get_dm_mujoco()
            if model_path.endswith(".mjb"):
                self.sim = dm_mujoco.Physics.from_binary_path(model_path)
            else:
                self.sim = dm_mujoco.Physics.from_xml_path(model_path)
            self.model = self.sim.model
            self._patch_mjlib_accessors(self.model, self.sim.data)

            self.renderer = DMRenderer(self.sim, camera_settings=camera_settings)
        else:  # Use mujoco_py
            mujoco_py = module.get_mujoco_py()
            self.model = mujoco_py.load_model_from_path(model_path)
            self.sim = mujoco_py.MjSim(self.model)
            self.renderer = MjPyRenderer(self.sim, camera_settings=camera_settings)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}
        self._rgb_array_res = rgb_array_res

        self.metadata = {
            "render.modes": ["human"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._did_see_sim_exception = False

        self.seed()

    def get_mjlib(self):
        """Returns an object that exposes the low-level MuJoCo API."""
        if self._use_dm_backend:
            return module.get_dm_mujoco().wrapper.mjbindings.mjlib
        else:
            return module.get_mujoco_py_mjlib()

    def _patch_mjlib_accessors(self, model, data):
        """Adds accessors to the DM Control objects to support mujoco_py API."""
        assert self._use_dm_backend
        mjlib = self.get_mjlib()

        def name2id(type_name, name):
            obj_id = mjlib.mj_name2id(
                model.ptr, mjlib.mju_str2Type(type_name.encode()), name.encode()
            )
            if obj_id < 0:
                raise ValueError('No {} with name "{}" exists.'.format(type_name, name))
            return obj_id

        if not hasattr(model, "body_name2id"):
            model.body_name2id = lambda name: name2id("body", name)

        if not hasattr(model, "geom_name2id"):
            model.geom_name2id = lambda name: name2id("geom", name)

        if not hasattr(model, "site_name2id"):
            model.site_name2id = lambda name: name2id("site", name)

        if not hasattr(model, "joint_name2id"):
            model.joint_name2id = lambda name: name2id("joint", name)

        if not hasattr(model, "actuator_name2id"):
            model.actuator_name2id = lambda name: name2id("actuator", name)

        if not hasattr(model, "camera_name2id"):
            model.camera_name2id = lambda name: name2id("camera", name)

        if not hasattr(data, "body_xpos"):
            data.body_xpos = data.xpos

        if not hasattr(data, "body_xquat"):
            data.body_xquat = data.xquat

        if not hasattr(data, "get_body_xpos"):
            data.get_body_xpos = lambda name: data.body_xpos[model.body_name2id(name)]

        if not hasattr(data, "get_body_xquat"):
            data.get_body_xquat = lambda name: data.body_xquat[model.body_name2id(name)]

        if not hasattr(data, "get_body_xmat"):
            # (TODO): verify this is correct reshape and make sure xmat is the right thing for body_xmat
            data.get_body_xmat = lambda name: data.xmat[
                model.body_name2id(name)
            ].reshape(3, 3)

        if not hasattr(data, "get_geom_xpos"):
            data.get_geom_xpos = lambda name: data.geom_xpos[model.geom_name2id(name)]

        if not hasattr(data, "get_geom_xquat"):
            data.get_geom_xquat = lambda name: data.geom_xquat[model.geom_name2id(name)]

        if not hasattr(data, "get_joint_qpos"):
            # (TODO): verify this is the correct index
            data.get_joint_qpos = lambda name: data.qpos[model.joint_name2id(name)]

        if not hasattr(data, "set_joint_qpos"):
            # (TODO): verify this is the correct index
            def set_joint_qpos(name, value):
                data.qpos[model.joint_name2id(name)] = value

            data.set_joint_qpos = lambda name, value: set_joint_qpos(name, value)

        if not hasattr(data, "get_site_xmat"):
            # (TODO): verify this is correct reshape
            data.get_site_xmat = lambda name: data.site_xmat[
                model.site_name2id(name)
            ].reshape(3, 3)

        if not hasattr(model, "get_joint_qpos_addr"):
            model.get_joint_qpos_addr = lambda name: model.joint_name2id(name)

        if not hasattr(data, "get_geom_xmat"):
            data.get_geom_xmat = lambda name: data.geom_xmat[
                model.geom_name2id(name)
            ].reshape(
                3, 3
            )  # (TODO): verify this is correct reshape

        if not hasattr(data, "get_mocap_pos"):
            data.get_mocap_pos = lambda name: data.mocap_pos[
                model.body_mocapid[model.body_name2id(name)]
            ]

        if not hasattr(data, "get_mocap_quat"):
            data.get_mocap_quat = lambda name: data.mocap_quat[
                model.body_mocapid[model.body_name2id(name)]
            ]

        if not hasattr(data, "set_mocap_pos"):

            def set_mocap_pos(name, value):
                data.mocap_pos[model.body_mocapid[model.body_name2id(name)]] = value

            data.set_mocap_pos = lambda name, value: set_mocap_pos(name, value)

        if not hasattr(data, "set_mocap_quat"):

            def set_mocap_quat(name, value):
                data.mocap_quat[model.body_mocapid[model.body_name2id(name)]] = value

            data.set_mocap_quat = lambda name, value: set_mocap_quat(name, value)

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        if self._use_dm_backend:
            self.sim.set_state(np.concatenate((qpos, qvel)))
        else:
            old_state = self.sim.get_state()
            new_state = mujoco_py.MjSimState(
                old_state.time, qpos, qvel, old_state.act, old_state.udd_state
            )
            self.sim.set_state(new_state)
        self.sim.forward()

    def render(
        self,
        mode="human",
        width=64,
        height=64,
    ):
        if mode == "human":
            self.renderer.render_to_window()
        elif mode == "rgb_array":
            return self.renderer.render_offscreen(
                width,
                height,
            )
        else:
            raise ValueError("mode can only be either 'human' or 'rgb_array'")


class SawyerMocapBaseDMBackendMetaworld(
    DMControlBackendMetaworldMujocoEnv, metaclass=abc.ABCMeta
):
    """
    Provides some commonly-shared functions for Sawyer Mujoco envs that use
    mocap for XYZ control.
    """

    mocap_low = np.array([-0.2, 0.5, 0.06])
    mocap_high = np.array([0.2, 0.7, 0.6])

    def __init__(self, model_name, frame_skip=5):
        DMControlBackendMetaworldMujocoEnv.__init__(
            self, model_name, frame_skip=frame_skip
        )
        self.reset_mocap_welds()

    def get_endeff_pos(self):
        return self.data.get_body_xpos("hand").copy()

    @property
    def tcp_center(self):
        """The COM of the gripper's 2 fingers

        Returns:
            (np.ndarray): 3-element position
        """
        right_finger_pos = self._get_site_pos("rightEndEffector")
        left_finger_pos = self._get_site_pos("leftEndEffector")
        tcp_center = (right_finger_pos + left_finger_pos) / 2.0
        return tcp_center

    def get_env_state(self):
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        state = joint_state, mocap_state
        return copy.deepcopy(state)

    def set_env_state(self, state):
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.data.set_mocap_pos("mocap", mocap_pos)
        self.data.set_mocap_quat("mocap", mocap_quat)
        self.sim.forward()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["model"]
        del state["sim"]
        del state["data"]
        mjb = self.model.get_mjb()
        return {"state": state, "mjb": mjb, "env_state": self.get_env_state()}

    def __setstate__(self, state):
        self.__dict__ = state["state"]
        self.model = mujoco_py.load_model_from_mjb(state["mjb"])
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.set_env_state(state["env_state"])

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                    )
        sim.forward()


class DMControlBackendMetaworldRobosuiteEnv(robosuite.environments.base.MujocoEnv):
    def __init__(self, *args, use_dm_backend=True, **kwargs):
        self._use_dm_backend = use_dm_backend
        super().__init__(*args, **kwargs)

    def _reset_internal(self):
        """Resets simulation internal configurations."""

        # create visualization screen or renderer
        if self.has_renderer and self.viewer is None and not self._use_dm_backend:
            self.viewer = MujocoPyRenderer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = (
                1 if self.render_collision_mesh else 0
            )
            self.viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0

            # hiding the overlay speeds up rendering significantly
            self.viewer.viewer._hide_overlay = True

            # make sure mujoco-py doesn't block rendering frames
            # (see https://github.com/StanfordVL/robosuite/issues/39)
            self.viewer.viewer._render_every_frame = True

            # Set the camera angle for viewing
            if self.render_camera is not None:
                self.viewer.set_camera(
                    camera_id=self.sim.model.camera_name2id(self.render_camera)
                )

        elif self.has_offscreen_renderer:
            if self._use_dm_backend:
                self.renderer = DMRenderer(self.sim, camera_settings={})
            else:
                if self.sim._render_context_offscreen is None:
                    render_context = mujoco_py.MjRenderContextOffscreen(
                        self.sim, device_id=self.render_gpu_device_id
                    )
                    self.sim.add_render_context(render_context)
                self.sim._render_context_offscreen.vopt.geomgroup[0] = (
                    1 if self.render_collision_mesh else 0
                )
                self.sim._render_context_offscreen.vopt.geomgroup[1] = (
                    1 if self.render_visual_mesh else 0
                )

        # additional housekeeping
        self.sim_state_initial = self.sim.get_state()
        self._setup_references()
        self.cur_time = 0
        self.timestep = 0
        self.done = False

        # Empty observation cache and reset all observables
        self._obs_cache = {}
        for observable in self._observables.values():
            observable.reset()

    def _create_camera_sensors(self, cam_name, cam_w, cam_h, cam_d, modality="image"):
        """
        Helper function to create sensors for a given camera. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            cam_name (str): Name of camera to create sensors for
            cam_w (int): Width of camera
            cam_h (int): Height of camera
            cam_d (bool): Whether to create a depth sensor as well
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given camera
                names (list): array of corresponding observable names
        """
        # Make sure we get correct convention
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

        # Create sensor information
        sensors = []
        names = []

        # Add camera observables to the dict
        rgb_sensor_name = f"{cam_name}_image"
        depth_sensor_name = f"{cam_name}_depth"

        @sensor(modality=modality)
        def camera_rgb(obs_cache):
            if self._use_dm_backend:
                img = self.renderer.render_offscreen(
                    cam_w,
                    cam_h,
                )
            else:
                img = self.sim.render(
                    camera_name=cam_name,
                    width=cam_w,
                    height=cam_h,
                    depth=cam_d,
                )
            if cam_d:
                rgb, depth = img
                obs_cache[depth_sensor_name] = np.expand_dims(
                    depth[::convention], axis=-1
                )
                return rgb[::convention]
            else:
                return img[::convention]

        sensors.append(camera_rgb)
        names.append(rgb_sensor_name)

        if cam_d:

            @sensor(modality=modality)
            def camera_depth(obs_cache):
                return (
                    obs_cache[depth_sensor_name]
                    if depth_sensor_name in obs_cache
                    else np.zeros((cam_h, cam_w, 1))
                )

            sensors.append(camera_depth)
            names.append(depth_sensor_name)

        return sensors, names

    def _initialize_sim(self, xml_string=None):
        """
        Creates a MjSim object and stores it in self.sim. If @xml_string is specified, the MjSim object will be created
        from the specified xml_string. Else, it will pull from self.model to instantiate the simulation

        Args:
            xml_string (str): If specified, creates MjSim object from this filepath
        """
        super()._initialize_sim(xml_string)
        if self._use_dm_backend:
            dm_mujoco = module.get_dm_mujoco()

            with io.StringIO() as string:
                string.write(ET.tostring(self.model.root, encoding="unicode"))
                self.sim = dm_mujoco.Physics.from_xml_string(string.getvalue())

            self.mjpy_model = self.sim.model
            self._patch_mjlib_accessors(self.mjpy_model, self.sim.data)

            self.renderer = DMRenderer(self.sim, camera_settings={})
        else:  # Use mujoco_py

            with io.StringIO() as string:
                string.write(ET.tostring(self.model.root, encoding="unicode"))
                from mujoco_py import load_model_from_xml

                self.mjpy_model = load_model_from_xml(string.getvalue())
            mujoco_py = module.get_mujoco_py()
            self.sim = mujoco_py.MjSim(self.mjpy_model)
            self.renderer = MjPyRenderer(self.sim, camera_settings={})

    def get_mjlib(self):
        """Returns an object that exposes the low-level MuJoCo API."""
        if self._use_dm_backend:
            return module.get_dm_mujoco().wrapper.mjbindings.mjlib
        else:
            return module.get_mujoco_py_mjlib()

    def _patch_mjlib_accessors(self, model, data):
        """Adds accessors to the DM Control objects to support mujoco_py API.
        obtained from https://github.com/openai/mujoco-py/blob/master/mujoco_py/generated/wrappers.pxi
        """
        assert self._use_dm_backend
        mjlib = self.get_mjlib()

        def name2id(type_name, name):
            obj_id = mjlib.mj_name2id(
                model.ptr, mjlib.mju_str2Type(type_name.encode()), name.encode()
            )
            if obj_id < 0:
                raise ValueError('No {} with name "{}" exists.'.format(type_name, name))
            return obj_id

        def id2name(type_name, id):
            obj_name = mjlib.mj_id2name(
                model.ptr, mjlib.mju_str2Type(type_name.encode()), id
            )
            return obj_name

        if not hasattr(model, "body_name2id"):
            model.body_name2id = lambda name: name2id("body", name)

        if not hasattr(model, "geom_name2id"):
            model.geom_name2id = lambda name: name2id("geom", name)

        if not hasattr(model, "geom_id2name"):
            model.geom_id2name = lambda id: id2name("geom", id)

        if not hasattr(model, "site_name2id"):
            model.site_name2id = lambda name: name2id("site", name)

        if not hasattr(model, "joint_name2id"):
            model.joint_name2id = lambda name: name2id("joint", name)

        if not hasattr(model, "actuator_name2id"):
            model.actuator_name2id = lambda name: name2id("actuator", name)

        if not hasattr(model, "camera_name2id"):
            model.camera_name2id = lambda name: name2id("camera", name)

        if not hasattr(model, "sensor_name2id"):
            model.sensor_name2id = lambda name: name2id("sensor", name)

        if not hasattr(data, "body_xpos"):
            data.body_xpos = data.xpos

        if not hasattr(data, "body_xquat"):
            data.body_xquat = data.xquat

        if not hasattr(data, "get_body_xpos"):
            data.get_body_xpos = lambda name: data.body_xpos[model.body_name2id(name)]

        if not hasattr(data, "get_body_xquat"):
            data.get_body_xquat = lambda name: data.body_xquat[model.body_name2id(name)]

        if not hasattr(data, "get_body_xmat"):
            # (TODO): verify this is correct reshape and make sure xmat is the right thing for body_xmat
            data.get_body_xmat = lambda name: data.xmat[
                model.body_name2id(name)
            ].reshape(3, 3)

        if not hasattr(data, "get_geom_xpos"):
            data.get_geom_xpos = lambda name: data.geom_xpos[model.geom_name2id(name)]

        if not hasattr(data, "get_geom_xquat"):
            data.get_geom_xquat = lambda name: data.geom_xquat[model.geom_name2id(name)]

        if not hasattr(data, "get_joint_qpos"):
            # (TODO): verify this is the correct index
            data.get_joint_qpos = lambda name: data.qpos[model.joint_name2id(name)]

        if not hasattr(data, "set_joint_qpos"):
            # (TODO): verify this is the correct index
            def set_joint_qpos(name, value):
                data.qpos[
                    model.joint_name2id(name) : model.joint_name2id(name)
                    + value.shape[0]
                ] = value

            data.set_joint_qpos = lambda name, value: set_joint_qpos(name, value)

        if not hasattr(data, "get_site_xmat"):
            # (TODO): verify this is correct reshape
            data.get_site_xmat = lambda name: data.site_xmat[
                model.site_name2id(name)
            ].reshape(3, 3)

        if not hasattr(model, "get_joint_qpos_addr"):
            model.get_joint_qpos_addr = lambda name: model.joint_name2id(name)

        if not hasattr(model, "get_joint_qvel_addr"):
            model.get_joint_qvel_addr = lambda name: model.joint_name2id(name)

        if not hasattr(data, "get_geom_xmat"):
            data.get_geom_xmat = lambda name: data.geom_xmat[
                model.geom_name2id(name)
            ].reshape(
                3, 3
            )  # (TODO): verify this is correct reshape

        if not hasattr(data, "get_mocap_pos"):
            data.get_mocap_pos = lambda name: data.mocap_pos[
                model.body_mocapid[model.body_name2id(name)]
            ]

        if not hasattr(data, "get_mocap_quat"):
            data.get_mocap_quat = lambda name: data.mocap_quat[
                model.body_mocapid[model.body_name2id(name)]
            ]

        if not hasattr(data, "set_mocap_pos"):

            def set_mocap_pos(name, value):
                data.mocap_pos[model.body_mocapid[model.body_name2id(name)]] = value

            data.set_mocap_pos = lambda name, value: set_mocap_pos(name, value)

        if not hasattr(data, "set_mocap_quat"):

            def set_mocap_quat(name, value):
                data.mocap_quat[model.body_mocapid[model.body_name2id(name)]] = value

            data.set_mocap_quat = lambda name, value: set_mocap_quat(name, value)

        def site_jacp():
            jacps = np.zeros((model.nsite, 3 * model.nv))
            for i, jacp in enumerate(jacps):
                jacp_view = jacp
                mjlib.mj_jacSite(model.ptr, data.ptr, jacp_view, None, i)
            return jacps

        def site_xvelp():
            jacp = site_jacp().reshape((model.nsite, 3, model.nv))
            xvelp = np.dot(jacp, data.qvel)
            return xvelp

        def site_jacr():
            jacrs = np.zeros((model.nsite, 3 * model.nv))
            for i, jacr in enumerate(jacrs):
                jacr_view = jacr
                mjlib.mj_jacSite(model.ptr, data.ptr, None, jacr_view, i)
            return jacrs

        def site_xvelr():
            jacr = site_jacr().reshape((model.nsite, 3, model.nv))
            xvelr = np.dot(jacr, data.qvel)
            return xvelr

        if not hasattr(data, "site_xvelp"):
            data.site_xvelp = site_xvelp()

        if not hasattr(data, "site_xvelr"):
            data.site_xvelr = site_xvelr()

        if not hasattr(data, "get_site_jacp"):
            # (TODO): verify this is correct reshape
            data.get_site_jacp = lambda name: site_jacp()[
                model.site_name2id(name)
            ].reshape(3, model.nv)

        if not hasattr(data, "get_site_jacr"):
            # (TODO): verify this is correct reshape
            data.get_site_jacr = lambda name: site_jacr()[
                model.site_name2id(name)
            ].reshape(3, model.nv)
