import pickle

import cv2
import gym
import mujoco_py
import numpy as np
import quaternion
import robosuite
from gym.spaces.box import Box
from metaworld.envs.mujoco.mujoco_env import _assert_task_is_set
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite_vices.controllers.arm_controller import PositionController

from rlkit.envs.dm_backend_wrappers import DMControlBackendMetaworldRobosuiteEnv
from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        gym.Wrapper.__init__(self, env)
        self._duration = duration
        self._elapsed_steps = 0
        self._max_episode_steps = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(
            action,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        self._step += 1
        self._elapsed_steps += 1
        if self._step >= self._duration:
            done = True
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        self._elapsed_steps = 0
        return self.env.reset()


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, amount):
        gym.Wrapper.__init__(self, env)
        self._amount = amount

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
            obs, reward, done, info = self.env.step(
                action,
                render_every_step=render_every_step,
                render_mode=render_mode,
                render_im_shape=render_im_shape,
            )
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class NormalizeActions(gym.Wrapper):
    def __init__(self, env, unused=None):
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

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        o, r, d, i = self.env.step(
            original,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        return o, r, d, i

    def reset(self):
        return self.env.reset()


class ImageUnFlattenWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = Box(
            0, 255, (3, self.env.imwidth, self.env.imheight), dtype=np.uint8
        )

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs = self.env.reset()
        return obs.reshape(-1, self.env.imwidth, self.env.imheight)

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        obs, reward, done, info = self.env.step(
            action,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
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

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs = self.env.reset()
        return obs.reshape(-1, self.env.imwidth, self.env.imheight).transpose(1, 2, 0)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return (
            obs.reshape(-1, self.env.imwidth, self.env.imheight).transpose(1, 2, 0),
            reward,
            done,
            info,
        )


class MetaworldWrapper(gym.Wrapper):
    def __init__(self, env, reward_type="dense"):
        super().__init__(env)
        self.reward_type = reward_type

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs = super().reset()
        return obs

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        self.set_render_every_step(render_every_step, render_mode, render_im_shape)
        o, r, d, i = self.env.step(
            action,
        )
        self.unset_render_every_step()
        new_i = {}
        for k, v in i.items():
            if v is not None:
                new_i[k] = v
        if self.reward_type == "dense":
            r = r
        elif self.reward_type == "sparse":
            r = i["success"]
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

    def save_image(self):
        img = (
            self._get_image().reshape(3, self.imwidth, self.imheight).transpose(1, 2, 0)
        )
        cv2.imwrite("test/" + type(self.env.env).__name__ + ".png", img)

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        o, r, d, i = self.env.step(
            action,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
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
        spaces["image"] = gym.spaces.Box(
            0, 255, (env.imwidth, env.imwidth, 3), dtype=np.uint8
        )
        self.observation_space = gym.spaces.Dict(spaces)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        o, r, d, i = self.env.step(
            action,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        return {"image": o}, r, d, i

    def reset(self):
        return {"image": self.env.reset()}


class IgnoreLastAction(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            np.concatenate((self.env.action_space.low, [0])),
            np.concatenate((self.env.action_space.high, [0])),
        )

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        return self.env.step(
            action[:-1],
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )


class GetObservationWrapper(gym.Wrapper):
    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observation(self):
        return self._get_obs()

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        return self.env.step(
            action,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )


class SawyerXYZEnvMetaworldPrimitives(SawyerXYZEnv):
    def reset_action_space(
        self,
        control_mode="end_effector",
        use_combined_action_space=True,
        action_scale=1 / 100,
        max_path_length=500,
    ):
        self.max_path_length = max_path_length
        self.action_scale = action_scale

        # primitives
        self.primitive_idx_to_name = {
            0: "move_delta_ee_pose",
            1: "top_x_y_grasp",
            2: "lift",
            3: "drop",
            4: "move_left",
            5: "move_right",
            6: "move_forward",
            7: "move_backward",
            8: "open_gripper",
            9: "close_gripper",
        }
        self.primitive_name_to_func = dict(
            move_delta_ee_pose=self.move_delta_ee_pose,
            top_x_y_grasp=self.top_x_y_grasp,
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
            top_x_y_grasp=[3, 4, 5],
            lift=6,
            drop=7,
            move_left=8,
            move_right=9,
            move_forward=10,
            move_backward=11,
            open_gripper=[],  # doesn't matter
            close_gripper=[],  # doesn't matter
        )
        self.max_arg_len = 12
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
            self.reset_mocap2body_xpos(self.sim)
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

        if self.control_mode == "vices":
            self.action_space = Box(-np.ones(10), np.ones(10))
            ctrl_ratio = 1.0
            control_range_pos = np.ones(3)
            kp_max = 10
            kp_max_abs_delta = 10
            kp_min = 0.1
            damping_max = 2
            damping_max_abs_delta = 1
            damping_min = 0.1
            use_delta_impedance = False
            initial_impedance_pos = 1
            initial_impedance_ori = 1
            initial_damping = 0.25
            control_freq = 1.0 * ctrl_ratio

            self.joint_index_vel = np.arange(7)
            self.controller = PositionController(
                control_range_pos,
                kp_max,
                kp_max_abs_delta,
                kp_min,
                damping_max,
                damping_max_abs_delta,
                damping_min,
                use_delta_impedance,
                initial_impedance_pos,
                initial_impedance_ori,
                initial_damping,
                control_freq=control_freq,
                interpolation="linear",
            )
            self.controller.update_model(
                self.sim, self.joint_index_vel, self.joint_index_vel
            )
        self.unset_render_every_step()

    def _reset_hand(self):
        if self.control_mode != "vices":
            super()._reset_hand()
        else:
            self.sim.data.qpos[:] = self.reset_qpos
            self.sim.forward()
            self.init_tcp = self.tcp_center

    def set_render_every_step(
        self,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        self.render_every_step = render_every_step
        self.render_mode = render_mode
        self.render_im_shape = render_im_shape

    def unset_render_every_step(self):
        self.render_every_step = False

    @_assert_task_is_set
    def step(
        self,
        action,
    ):
        a = np.clip(action, -1.0, 1.0)
        if self.control_mode in [
            "joint_position",
            "joint_velocity",
            "torque",
            "end_effector",
            "vices",
        ]:
            if self.control_mode == "end_effector":
                self.set_xyz_action(a[:3])
                self.do_simulation([a[-1], -a[-1]])
            elif self.control_mode == "vices":
                for i in range(int(self.controller.interpolation_steps)):
                    self.controller.update_model(
                        self.sim, self.joint_index_vel, self.joint_index_vel
                    )
                    a = self.controller.action_to_torques(action[:-1], i == 0)
                    act = np.zeros(9)
                    act[-1] = -action[-1]
                    act[-2] = action[-1]
                    act[:7] = a
                    self.do_simulation(act, n_frames=1)
            stats = [0, 0]
        else:
            self.img_array = []
            stats = self.act(
                a,
                render_every_step=self.render_every_step,
                render_mode=self.render_mode,
                render_im_shape=self.render_im_shape,
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
        if self.control_mode == "primitives":
            reward = stats[0]
            info["success"] = float(stats[1] > 0)
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
        self.sim.data.ctrl[0] = action[0]
        self.sim.data.ctrl[1] = action[1]

    def mocap_set_action(self, sim, action):
        if sim.model.nmocap > 0:
            action, _ = np.split(action, (sim.model.nmocap * 7,))
            action = action.reshape(sim.model.nmocap, 7)

            pos_delta = action[:, :3]
            quat_delta = action[:, 3:]
            self.reset_mocap2body_xpos(sim)
            new_mocap_pos = self.data.mocap_pos + pos_delta[None]
            new_mocap_quat = self.data.mocap_quat + quat_delta[None]

            new_mocap_pos[0, :] = np.clip(
                new_mocap_pos[0, :],
                self.mocap_low,
                self.mocap_high,
            )
            self.data.set_mocap_pos("mocap", new_mocap_pos)
            self.data.set_mocap_quat("mocap", new_mocap_quat)

    def _set_action(self, action):
        assert action.shape == (9,)

        action = action.copy()
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:7], action[7:9]

        pos_ctrl *= 0.05
        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, rot_ctrl])

        # Apply action to simulation.
        self.mocap_set_action(self.sim, action)
        self.ctrl_set_action(self.sim, gripper_ctrl)

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

    def close_gripper(
        self,
        unused=None,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        total_reward, total_success = 0, 0
        for _ in range(300):
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
            r, info = self.evaluate_state(self._get_obs(), [0, 0, 0, -1])
            total_reward += r
            total_success += info["success"]
        return np.array((total_reward, total_success))

    def open_gripper(
        self,
        unused=None,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        total_reward, total_success = 0, 0
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
            r, info = self.evaluate_state(self._get_obs(), [0, 0, 0, 1])
            total_reward += r
            total_success += info["success"]
        return np.array((total_reward, total_success))

    def goto_pose(
        self,
        pose,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
        grasp=False,
    ):
        total_reward, total_success = 0, 0
        for _ in range(300):
            delta = pose - self.get_endeff_pos()
            gripper = self.sim.data.qpos[8:10]
            if grasp:
                gripper = [1, -1]
            self._set_action(
                np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, 0.0, *gripper])
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
            r, info = self.evaluate_state(self._get_obs(), [*delta, 0])
            total_reward += r
            total_success += info["success"]
        return np.array((total_reward, total_success))

    def top_x_y_grasp(
        self,
        xyz,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        x_dist, y_dist, z_dist = xyz
        stats = self.open_gripper(
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        stats += self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, y_dist, 0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        stats += self.goto_pose(
            self.get_endeff_pos() + np.array([x_dist, 0.0, 0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        stats += self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, 0, z_dist]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        stats += self.close_gripper(
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        return stats

    def move_delta_ee_pose(
        self,
        pose,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        stats = self.goto_pose(
            self.get_endeff_pos() + pose,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        return stats

    def lift(
        self,
        z_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        z_dist = np.maximum(z_dist, 0.0)
        stats = self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, 0.0, z_dist]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
            grasp=True,
        )
        return stats

    def drop(
        self,
        z_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        z_dist = np.maximum(z_dist, 0.0)
        stats = self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, 0.0, -z_dist]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
            grasp=True,
        )
        return stats

    def move_left(
        self,
        x_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        x_dist = np.maximum(x_dist, 0.0)
        stats = self.goto_pose(
            self.get_endeff_pos() + np.array([-x_dist, 0.0, 0.0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
            grasp=True,
        )
        return stats

    def move_right(
        self,
        x_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        x_dist = np.maximum(x_dist, 0.0)
        stats = self.goto_pose(
            self.get_endeff_pos() + np.array([x_dist, 0.0, 0.0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
            grasp=True,
        )
        return stats

    def move_forward(
        self,
        y_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        y_dist = np.maximum(y_dist, 0.0)
        stats = self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, y_dist, 0.0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
            grasp=True,
        )
        return stats

    def move_backward(
        self,
        y_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        y_dist = np.maximum(y_dist, 0.0)
        stats = self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, -y_dist, 0.0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
            grasp=True,
        )
        return stats

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
            stats = primitive(
                primitive_action,
                render_every_step=render_every_step,
                render_mode=render_mode,
                render_im_shape=render_im_shape,
            )
        return stats

    # (TODO): fix this for dm control backend
    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


class RobosuiteWrapper(GymWrapper):
    def __init__(
        self,
        env,
        keys=None,
        control_mode="robosuite",
        use_combined_action_space=True,
        action_scale=1,
        max_path_length=200,
    ):
        super().__init__(
            env,
            keys=keys,
        )
        if hasattr(self.env, "reset_action_space"):
            self.env.reset_action_space(
                control_mode, use_combined_action_space, action_scale, max_path_length
            )
            if self.control_mode == "primitives":
                self.action_space = self.env.action_space
        self.imwidth = 64
        self.imheight = 64
        self.image_shape = (3, self.imwidth, self.imheight)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs = super().reset()
        if self.env.use_camera_obs:
            obs = obs.reshape(64, 64, 3)[:, :, ::-1].transpose(2, 0, 1).flatten()
        return obs

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        if hasattr(self.env, "set_render_every_step"):
            self.env.set_render_every_step(
                render_every_step, render_mode, render_im_shape
            )
        o, r, d, i = super().step(action)
        if hasattr(self.env, "unset_render_every_step"):
            self.env.unset_render_every_step()
        new_i = {}
        for k, v in i.items():
            if v is not None:
                new_i[k] = v
        if self.env.use_camera_obs:
            o = o.reshape(64, 64, 3)[:, :, ::-1].transpose(2, 0, 1).flatten()
        return o, r, d, new_i

    def __getattr__(self, name):
        return getattr(self.env, name)


class NormalizeBoxEnvFixed(NormalizedBoxEnv):
    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)
        wrapped_step = self._wrapped_env.step(
            scaled_action,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info


class RobosuitePrimitives(DMControlBackendMetaworldRobosuiteEnv):
    def set_render_every_step(
        self,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        self.render_every_step = render_every_step
        self.render_mode = render_mode
        self.render_im_shape = render_im_shape

    def unset_render_every_step(self):
        self.render_every_step = False

    def reset_action_space(
        self,
        control_mode="robosuite",
        use_combined_action_space=True,
        action_scale=1,
        max_path_length=200,
    ):
        self.max_path_length = max_path_length
        self.action_scale = action_scale

        # primitives
        self.primitive_idx_to_name = {
            0: "move_delta_ee_pose",
            1: "top_x_y_grasp",
            2: "lift",
            3: "drop",
            4: "move_left",
            5: "move_right",
            6: "move_forward",
            7: "move_backward",
            8: "open_gripper",
            9: "close_gripper",
        }
        self.primitive_name_to_func = dict(
            move_delta_ee_pose=self.move_delta_ee_pose,
            top_x_y_grasp=self.top_x_y_grasp,
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
            top_x_y_grasp=[3, 4, 5],
            lift=6,
            drop=7,
            move_left=8,
            move_right=9,
            move_forward=10,
            move_backward=11,
            open_gripper=[],  # doesn't matter
            close_gripper=[],  # doesn't matter
        )
        self.max_arg_len = 12
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

    def step(self, action):
        """
        Takes a step in simulation with control command @action.

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information

        Raises:
            ValueError: [Steps past episode termination]

        """
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1
        if self.control_mode == "robosuite":
            # Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
            # multiple torque commands in between new high level action commands. Therefore, we need to denote via
            # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
            # or an actual policy update
            policy_step = True

            # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
            # (as defined by the control frequency specified at the environment level)
            for i in range(int(self.control_timestep / self.model_timestep)):
                self.sim.forward()
                self._pre_action(action, policy_step)
                self.sim.step()
                self._update_observables()
                policy_step = False
                if self.render_every_step:
                    self.render()
            stats = [0, 0]
        else:
            self.img_array = []
            stats = self.act(
                action,
                render_every_step=self.render_every_step,
                render_mode=self.render_mode,
                render_im_shape=self.render_im_shape,
            )

        # Note: this is done all at once to avoid floating point inaccuracies
        self.cur_time += self.control_timestep

        reward, done, info = self._post_action(action)
        if self.control_mode == "primitives":
            reward = stats[0]
            info["success"] = float(stats[1] > 0)
        else:
            info["success"] = float(self._check_success())
        return self._get_observations(), reward, done, info

    def render(self, render_mode="human", imwidth=64, imheight=64):
        if render_mode == "human":
            self.renderer.render_to_window()
        else:
            img = self.renderer.render_offscreen(
                imwidth,
                imheight,
            )
            return img

    def close_gripper(
        self,
        unused=None,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        total_reward, total_success = 0, 0
        for _ in range(300):
            action = [0, 0, 0, 0, 0, 0, 1]
            self.robots[0].control(action, policy_step=False)
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
            r = self.reward(action)
            total_reward += r
            total_success += float(self._check_success())
        return np.array((total_reward, total_success))

    def open_gripper(
        self,
        unused=None,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        total_reward, total_success = 0, 0
        for _ in range(300):
            action = [0, 0, 0, 0, 0, 0, -1]
            self.robots[0].control(action, policy_step=False)
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
            r = self.reward(action)
            total_reward += r
            total_success += float(self._check_success())
        return np.array((total_reward, total_success))

    def goto_pose(
        self,
        pose,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
        grasp=False,
    ):
        total_reward, total_success = 0, 0
        prev_delta = np.zeros_like(pose)
        for _ in range(100):
            delta = pose - self._eef_xpos
            if grasp:
                gripper = 1
            else:
                gripper = 0
            action = [*delta, 0, 0, 0, gripper]
            if all(np.abs(delta - prev_delta) < 1e-4):
                break
            policy_step = True
            prev_delta = delta
            for i in range(int(self.control_timestep / self.model_timestep)):
                self.sim.forward()
                self._pre_action(action, policy_step)
                self.sim.step()
                policy_step = False
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
                r = self.reward(action)
                total_reward += r
                total_success += float(self._check_success())
        return np.array((total_reward, total_success))

    def top_x_y_grasp(
        self,
        xyz,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        x_dist, y_dist, z_dist = xyz
        stats = self.open_gripper(
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        stats += self.goto_pose(
            self._eef_xpos + np.array([y_dist, 0, 0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        stats += self.goto_pose(
            self._eef_xpos + np.array([0, x_dist, 0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        stats += self.drop(
            z_dist,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        stats += self.close_gripper(
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        return stats

    def move_delta_ee_pose(
        self,
        pose,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        stats = self.goto_pose(
            self._eef_xpos + pose,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        return stats

    def lift(
        self,
        z_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        z_dist = np.maximum(z_dist, 0.0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([0.0, 0.0, z_dist]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
            grasp=True,
        )
        return stats

    def drop(
        self,
        z_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        z_dist = np.maximum(z_dist, 0.0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([0.0, 0.0, -z_dist]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
            grasp=True,
        )
        return stats

    def move_left(
        self,
        x_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        x_dist = np.maximum(x_dist, 0.0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([0, -x_dist, 0.0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
            grasp=True,
        )
        return stats

    def move_right(
        self,
        x_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        x_dist = np.maximum(x_dist, 0.0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([0, x_dist, 0.0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
            grasp=True,
        )
        return stats

    def move_forward(
        self,
        y_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        y_dist = np.maximum(y_dist, 0.0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([y_dist, 0, 0.0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
            grasp=True,
        )
        return stats

    def move_backward(
        self,
        y_dist,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        y_dist = np.maximum(y_dist, 0.0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([-y_dist, 0, 0.0]),
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
            grasp=True,
        )
        return stats

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
            stats = primitive(
                primitive_action,
                render_every_step=render_every_step,
                render_mode=render_mode,
                render_im_shape=render_im_shape,
            )
        return stats

    def get_idx_from_primitive_name(self, primitive_name):
        for idx, pn in self.primitive_idx_to_name.items():
            if pn == primitive_name:
                return idx
