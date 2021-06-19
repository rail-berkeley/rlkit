import gym
import mujoco_py
import numpy as np
from d4rl.kitchen.adept_envs.simulation.renderer import DMRenderer
from gym import spaces
from gym.spaces.box import Box
from metaworld.envs.mujoco.mujoco_env import _assert_task_is_set
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv
from robosuite.wrappers.gym_wrapper import GymWrapper

try:
    from robosuite_vices.controllers.arm_controller import PositionController
except:
    pass

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

    def _get_image(self):
        if hasattr(self.env, "_use_dm_backend"):
            img = self.env.render(
                mode="rgb_array", imwidth=self.imwidth, imheight=self.imheight
            )
        else:
            img = self.env.sim.render(
                imwidth=self.imwidth,
                imheight=self.imheight,
            )

        img = img.transpose(2, 0, 1).flatten()
        return img

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
        self.env.imwdith = imwidth
        self.env.imheight = imheight
        self.imwidth = imwidth
        self.imheight = imheight
        self.observation_space = Box(
            0, 255, (3 * self.imwidth * self.imheight,), dtype=np.uint8
        )
        self.image_shape = (3, self.imwidth, self.imheight)
        self.reward_scale = reward_scale

    def _get_image(self):
        if hasattr(self.env, "_use_dm_backend"):
            img = self.env.render(
                mode="rgb_array", imwidth=self.imwidth, imheight=self.imheight
            )
        else:
            img = self.env.sim.render(
                imwidth=self.imwidth,
                imheight=self.imheight,
            )

        img = img.transpose(2, 0, 1).flatten()
        return img

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
        o = self._get_image()
        r = self.reward_scale * r
        new_i = {}
        for k, v in i.items():
            if v is not None:
                new_i[k] = v
        return o, r, d, new_i

    def reset(self):
        super().reset()
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
    def reset_camera(self, camera_settings):
        if camera_settings is None:
            camera_settings = {}
        self.renderer = DMRenderer(self.sim, camera_settings=camera_settings)

    def reset_action_space(
        self,
        control_mode="end_effector",
        action_scale=1 / 100,
        max_path_length=500,
        camera_settings=None,
    ):
        self.reset_camera(camera_settings)
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
        if self.control_mode == "primitives":
            action_space_low = -1 * np.ones(self.max_arg_len)
            action_space_high = np.ones(self.max_arg_len)
            self.reset_mocap2body_xpos(self.sim)
            act_lower_primitive = np.zeros(self.num_primitives)
            act_upper_primitive = np.ones(self.num_primitives)
            act_lower = np.concatenate((act_lower_primitive, action_space_low))
            act_upper = np.concatenate(
                (
                    act_upper_primitive,
                    action_space_high,
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
            stats = self.act(a)

        self.curr_path_length += 1

        for site in self._target_site_config:
            self._set_pos_site(*site)

        if self._did_see_sim_exception:
            return (
                self._last_stable_obs,
                0.0,
                False,
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
            return self._last_stable_obs

        reward, info = self.evaluate_state(self._last_stable_obs, action)
        if self.control_mode == "primitives":
            reward = stats[0]
            info["success"] = float(stats[1] > 0)
        return self._last_stable_obs, reward, False, info

    def _get_site_pos(self, siteName):
        return self.data.site_xpos[self.model.site_name2id(siteName)]

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

    def call_render_every_step(self):
        if self.render_every_step:
            if self.render_mode == "rgb_array":
                self.img_array.append(
                    self.render(
                        self.render_mode,
                        self.render_im_shape[0],
                        self.render_im_shape[1],
                    )
                )
            else:
                self.render(
                    self.render_mode,
                    self.render_im_shape[0],
                    self.render_im_shape[1],
                )

    def close_gripper(self, unused=None):
        total_reward, total_success = 0, 0
        for _ in range(300):
            self._set_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, -1]))
            self.data.set_mocap_quat("mocap", np.array([1, 0, 1, 0]))
            self.sim.step()
            self.call_render_every_step()
            r, info = self.evaluate_state(self._get_obs(), [0, 0, 0, -1])
            total_reward += r
            total_success += info["success"]
        return np.array((total_reward, total_success))

    def open_gripper(self, unused=None):
        total_reward, total_success = 0, 0
        for _ in range(200):
            self._set_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, 1]))
            self.data.set_mocap_quat("mocap", np.array([1, 0, 1, 0]))
            self.sim.step()
            self.call_render_every_step()
            r, info = self.evaluate_state(self._get_obs(), [0, 0, 0, 1])
            total_reward += r
            total_success += info["success"]
        return np.array((total_reward, total_success))

    def goto_pose(self, pose, grasp=True):
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
            self.call_render_every_step()
            r, info = self.evaluate_state(self._get_obs(), [*delta, 0])
            total_reward += r
            total_success += info["success"]
        return np.array((total_reward, total_success))

    def top_x_y_grasp(self, xyz):
        x_dist, y_dist, z_dist = xyz
        stats = self.open_gripper()
        stats += self.goto_pose(self.get_endeff_pos() + np.array([0.0, y_dist, 0]))
        stats += self.goto_pose(self.get_endeff_pos() + np.array([x_dist, 0.0, 0]))
        stats += self.goto_pose(self.get_endeff_pos() + np.array([0.0, 0, z_dist]))
        stats += self.close_gripper()
        return stats

    def move_delta_ee_pose(self, pose):
        stats = self.goto_pose(self.get_endeff_pos() + pose)
        return stats

    def lift(self, z_dist):
        z_dist = np.maximum(z_dist, 0.0)
        stats = self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, 0.0, z_dist]), grasp=True
        )
        return stats

    def drop(self, z_dist):
        z_dist = np.maximum(z_dist, 0.0)
        stats = self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, 0.0, -z_dist]), grasp=True
        )
        return stats

    def move_left(self, x_dist):
        x_dist = np.maximum(x_dist, 0.0)
        stats = self.goto_pose(
            self.get_endeff_pos() + np.array([-x_dist, 0.0, 0.0]), grasp=True
        )
        return stats

    def move_right(self, x_dist):
        x_dist = np.maximum(x_dist, 0.0)
        stats = self.goto_pose(
            self.get_endeff_pos() + np.array([x_dist, 0.0, 0.0]), grasp=True
        )
        return stats

    def move_forward(self, y_dist):
        y_dist = np.maximum(y_dist, 0.0)
        stats = self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, y_dist, 0.0]), grasp=True
        )
        return stats

    def move_backward(self, y_dist):
        y_dist = np.maximum(y_dist, 0.0)
        stats = self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, -y_dist, 0.0]), grasp=True
        )
        return stats

    def break_apart_action(self, a):
        broken_a = {}
        for k, v in self.primitive_name_to_action_idx.items():
            broken_a[k] = a[v]
        return broken_a

    def act(self, a):
        a = np.clip(a, self.action_space.low, self.action_space.high)
        a = a * self.action_scale
        primitive_idx, primitive_args = (
            np.argmax(a[: self.num_primitives]),
            a[self.num_primitives :],
        )
        primitive_name = self.primitive_idx_to_name[primitive_idx]
        primitive_name_to_action_dict = self.break_apart_action(primitive_args)
        primitive_action = primitive_name_to_action_dict[primitive_name]
        primitive = self.primitive_name_to_func[primitive_name]
        stats = primitive(
            primitive_action,
        )
        return stats

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


class RobosuiteWrapper(GymWrapper):
    def __init__(
        self,
        env,
        reset_action_space_kwargs,
        keys=None,
    ):
        super().__init__(
            env,
            keys=keys,
        )
        if hasattr(self.env, "reset_action_space"):
            self.env.reset_action_space(**reset_action_space_kwargs)
            if self.control_mode == "primitives" or self.control_mode == "vices":
                self.action_space = self.env.action_space
        self.image_shape = (3, self.imwidth, self.imheight)
        self.imlength = self.imwidth * self.imheight
        self.imlength *= 3
        self.observation_space = spaces.Box(0, 255, (self.imlength,), dtype=np.uint8)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        o = super().reset()
        o = self.env.render(
            render_mode="rgb_array", imheight=self.imheight, imwidth=self.imwidth
        )
        o = (
            o.reshape(self.imwidth, self.imheight, 3)[:, :, ::-1]
            .transpose(2, 0, 1)
            .flatten()
        )
        return o

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        self.env.set_render_every_step(render_every_step, render_mode, render_im_shape)
        o, r, d, i = super().step(action)
        o = self.env.render(
            render_mode="rgb_array", imheight=self.imheight, imwidth=self.imwidth
        )
        self.env.unset_render_every_step()
        new_i = {}
        for k, v in i.items():
            if v is not None:
                new_i[k] = v
        o = (
            o.reshape(self.imwidth, self.imheight, 3)[:, :, ::-1]
            .transpose(2, 0, 1)
            .flatten()
        )
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
        action_scale=1,
        max_path_length=200,
        camera_settings=None,
        workspace_low=(),
        workspace_high=(),
        imwidth=64,
        imheight=64,
        go_to_pose_iterations=100,
    ):
        self.imwidth = imwidth
        self.imheight = imheight
        self.workspace_low = np.array(workspace_low)
        self.workspace_high = np.array(workspace_high)
        if camera_settings is None:
            camera_settings = {}
        self.camera_settings = camera_settings
        self.max_path_length = max_path_length
        self.action_scale = action_scale
        self.go_to_pose_iterations = go_to_pose_iterations

        # primitives
        self.primitive_idx_to_name = {
            0: "move_delta_ee_pose",
            1: "top_grasp",
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
            top_grasp=self.top_grasp,
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
            top_grasp=3,
            lift=4,
            drop=5,
            move_left=6,
            move_right=7,
            move_forward=8,
            move_backward=9,
            open_gripper=[],  # doesn't matter
            close_gripper=[],  # doesn't matter
        )
        self.max_arg_len = 10

        self.num_primitives = len(self.primitive_name_to_func)
        self.control_mode = control_mode

        if self.control_mode == "primitives":
            action_space_low = -1 * np.ones(self.max_arg_len)
            action_space_high = np.ones(self.max_arg_len)
            act_lower_primitive = np.zeros(self.num_primitives)
            act_upper_primitive = np.ones(self.num_primitives)
            act_lower = np.concatenate((act_lower_primitive, action_space_low))
            act_upper = np.concatenate(
                (
                    act_upper_primitive,
                    action_space_high,
                )
            )
            self.action_space = Box(act_lower, act_upper, dtype=np.float32)
        elif self.control_mode == "vices":
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
                self.sim,
                self.joint_index_vel,
                self.joint_index_vel,
                id_name="robot0_right_hand",
            )

    def step(self, action):
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1
        if self.control_mode == "robosuite":
            policy_step = True
            target_pos = action[:3] + self._eef_xpos
            target_pos = np.clip(target_pos, self.workspace_low, self.workspace_high)
            action[:3] = target_pos - self._eef_xpos
            action[3:6] = 0

            for i in range(int(self.control_timestep / self.model_timestep)):
                self.sim.forward()
                self._pre_action(action, policy_step)
                self.sim.step()
                self._update_observables()
                policy_step = False
                if self.render_every_step:
                    self.render()
            stats = [0, 0]
        elif self.control_mode == "vices":
            for i in range(int(self.controller.interpolation_steps)):
                self.controller.update_model(
                    self.sim,
                    self.joint_index_vel,
                    self.joint_index_vel,
                    id_name="robot0_right_hand",
                )
                a = self.controller.action_to_torques(action[:-1], i == 0)
                act = np.zeros(9)
                act[-1] = -action[-1]
                act[-2] = action[-1]
                act[:7] = a
                self.sim.data.ctrl[:] = act
                self.sim.step()
                self._update_observables()

            stats = [0, 0]
        else:
            self.img_array = []
            stats = self.act(action)
            self._update_observables()

        self.cur_time += self.control_timestep

        reward, done, info = self._post_action(action)
        if self.control_mode == "primitives":
            reward = float(stats[1] > 0)
            info["success"] = float(stats[1] > 0)
        else:
            info["success"] = float(self._check_success())
        return self._get_observations(force_update=True), reward, done, info

    def render(self, render_mode="human", imwidth=64, imheight=64):
        if self._use_dm_backend:
            if render_mode == "human":
                self.renderer.render_to_window()
            else:
                img = self.renderer.render_offscreen(
                    imwidth,
                    imheight,
                )
                return img
        else:
            super().render()

    def call_render_every_step(self):
        if self.render_every_step:
            if self.render_mode == "rgb_array":
                self.img_array.append(
                    self.render(
                        self.render_mode,
                        self.render_im_shape[0],
                        self.render_im_shape[1],
                    )
                )
            else:
                self.render(
                    self.render_mode,
                    self.render_im_shape[0],
                    self.render_im_shape[1],
                )

    def close_gripper(self, unused=None):
        total_reward, total_success = 0, 0
        for _ in range(150):
            action = [0, 0, 0, 0, 0, 0, 1]
            self.robots[0].control(action, policy_step=False)
            self.sim.step()
            self.call_render_every_step()
            r = self.reward(action)
            total_reward += r
            total_success += float(self._check_success())
        return np.array((total_reward, total_success))

    def open_gripper(self, unused=None):
        total_reward, total_success = 0, 0
        for _ in range(300):
            action = [0, 0, 0, 0, 0, 0, -1]
            self.robots[0].control(action, policy_step=False)
            self.sim.step()
            self.call_render_every_step()
            r = self.reward(action)
            total_reward += r
            total_success += float(self._check_success())
        return np.array((total_reward, total_success))

    def goto_pose(self, pose, grasp=False):
        total_reward, total_success = 0, 0
        prev_delta = np.zeros_like(pose)
        pose = np.clip(pose, self.workspace_low, self.workspace_high)
        for _ in range(self.go_to_pose_iterations):
            delta = pose - self._eef_xpos
            if grasp:
                gripper = 1
            else:
                gripper = -1
            action = [*delta, 0, 0, 0, gripper]
            if np.allclose(delta - prev_delta, 1e-4):
                break
            policy_step = True
            prev_delta = delta
            for i in range(int(self.control_timestep / self.model_timestep)):
                self.sim.forward()
                self._pre_action(action, policy_step)
                self.sim.step()
                policy_step = False
                self.call_render_every_step()
                self.cur_time += self.control_timestep
                r = self.reward(action)
                total_reward += r
                total_success += float(self._check_success())
        return np.array((total_reward, total_success))

    def top_grasp(self, z_down):
        stats = self.goto_pose(
            self._eef_xpos + np.array([0, 0, -np.abs(z_down)]), grasp=False
        )
        stats += self.close_gripper()
        return stats

    def move_delta_ee_pose(self, pose):
        stats = self.goto_pose(self._eef_xpos + pose, grasp=True)
        return stats

    def lift(self, z_dist):
        z_dist = np.maximum(z_dist, 0.0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([0.0, 0.0, z_dist]), grasp=True
        )
        return stats

    def drop(self, z_dist):
        z_dist = np.maximum(z_dist, 0.0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([0.0, 0.0, -z_dist]), grasp=True
        )
        return stats

    def move_left(self, x_dist):
        x_dist = np.maximum(x_dist, 0.0)
        stats = self.goto_pose(self._eef_xpos + np.array([0, -x_dist, 0.0]), grasp=True)
        return stats

    def move_right(self, x_dist):
        x_dist = np.maximum(x_dist, 0.0)
        stats = self.goto_pose(self._eef_xpos + np.array([0, x_dist, 0.0]), grasp=True)
        return stats

    def move_forward(self, y_dist):
        y_dist = np.maximum(y_dist, 0.0)
        stats = self.goto_pose(self._eef_xpos + np.array([y_dist, 0, 0.0]), grasp=True)
        return stats

    def move_backward(self, y_dist):
        y_dist = np.maximum(y_dist, 0.0)
        stats = self.goto_pose(self._eef_xpos + np.array([-y_dist, 0, 0.0]), grasp=True)
        return stats

    def break_apart_action(self, a):
        broken_a = {}
        for k, v in self.primitive_name_to_action_idx.items():
            broken_a[k] = a[v]
        return broken_a

    def act(self, a):
        a = np.clip(a, self.action_space.low, self.action_space.high)
        a = a * self.action_scale
        primitive_idx, primitive_args = (
            np.argmax(a[: self.num_primitives]),
            a[self.num_primitives :],
        )
        primitive_name = self.primitive_idx_to_name[primitive_idx]
        primitive_name_to_action_dict = self.break_apart_action(primitive_args)
        primitive_action = primitive_name_to_action_dict[primitive_name]
        primitive = self.primitive_name_to_func[primitive_name]
        stats = primitive(primitive_action)
        return stats

    def get_idx_from_primitive_name(self, primitive_name):
        for idx, pn in self.primitive_idx_to_name.items():
            if pn == primitive_name:
                return idx
