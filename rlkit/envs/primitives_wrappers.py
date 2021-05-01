import pickle
import cv2
import gym
import mujoco_py
import numpy as np
import quaternion
from gym.spaces.box import Box
from metaworld.envs.mujoco.mujoco_env import _assert_task_is_set
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        gym.Wrapper.__init__(self, env)
        self._duration = duration
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
    def __init__(self, env, reward_type="dense"):
        super().__init__(env)
        self._elapsed_steps = 0
        self.reward_type = reward_type

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        obs = super().reset()
        self._elapsed_steps = 0
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
        self._elapsed_steps += 1
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

        img = img[:, :, ::-1].transpose(2, 0, 1).flatten()
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


class IgnoreLastAction(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(np.concatenate((self.env.action_space.low, [0])), np.concatenate((self.env.action_space.high, [0])))

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
        self.unset_render_every_step()

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
        ]:
            if self.control_mode == "end_effector":
                self.set_xyz_action(a[:3])
                self.do_simulation([a[-1], -a[-1]])
        else:
            self.img_array = []
            self.act(
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
