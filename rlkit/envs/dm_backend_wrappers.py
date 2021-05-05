import abc
import io
import xml.etree.ElementTree as ET
from os import path

import mujoco_py
import numpy as np
import robosuite
from d4rl.kitchen.adept_envs.simulation import module
from d4rl.kitchen.adept_envs.simulation.renderer import DMRenderer, MjPyRenderer
from metaworld.envs.mujoco.mujoco_env import MujocoEnv
from robosuite.utils import macros
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
from robosuite.utils.mujoco_py_renderer import MujocoPyRenderer
from robosuite.utils.observables import sensor


def patch_mjlib_accessors(mjlib, model, data):
    """Adds accessors to the DM Control objects to support mujoco_py API.
    obtained from https://github.com/openai/mujoco-py/blob/master/mujoco_py/generated/wrappers.pxi
    """

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
        data.get_body_xmat = lambda name: data.xmat[model.body_name2id(name)].reshape(
            3, 3
        )

    if not hasattr(data, "get_geom_xpos"):
        data.get_geom_xpos = lambda name: data.geom_xpos[model.geom_name2id(name)]

    if not hasattr(data, "get_geom_xquat"):
        data.get_geom_xquat = lambda name: data.geom_xquat[model.geom_name2id(name)]

    if not hasattr(data, "get_joint_qpos"):
        data.get_joint_qpos = lambda name: data.qpos[model.joint_name2id(name)]

    if not hasattr(data, "set_joint_qpos"):

        def set_joint_qpos(name, value):
            data.qpos[
                model.joint_name2id(name) : model.joint_name2id(name) + value.shape[0]
            ] = value

        data.set_joint_qpos = lambda name, value: set_joint_qpos(name, value)

    if not hasattr(data, "get_site_xmat"):
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
        ].reshape(3, 3)

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
        data.get_site_jacp = lambda name: site_jacp()[model.site_name2id(name)].reshape(
            3, model.nv
        )

    if not hasattr(data, "get_site_jacr"):
        data.get_site_jacr = lambda name: site_jacr()[model.site_name2id(name)].reshape(
            3, model.nv
        )


class DMControlBackendMetaworldMujocoEnv(MujocoEnv):
    def __init__(self, model_path, frame_skip, rgb_array_res=(640, 480)):
        if not path.exists(model_path):
            raise IOError("File %s does not exist" % model_path)
        self.frame_skip = frame_skip
        self._use_dm_backend = True
        camera_settings = {
            "distance": 1.878359835328275,
            "lookat": [0.16854934, 0.27084485, 0.23161897],
            "azimuth": 141.328125,
            "elevation": -53.203125160653144,
        }
        if self._use_dm_backend:
            dm_mujoco = module.get_dm_mujoco()
            if model_path.endswith(".mjb"):
                self.sim = dm_mujoco.Physics.from_binary_path(model_path)
            else:
                self.sim = dm_mujoco.Physics.from_xml_path(model_path)
            self.model = self.sim.model
            patch_mjlib_accessors(self.get_mjlib(), self.model, self.sim.data)

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
            )[:, :, ::-1]
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
        camera_settings = {}
        if self.has_renderer and self.viewer is None:
            if self._use_dm_backend:
                self.viewer = DMRenderer(
                    self.sim,
                    clear_geom_group_0=True,
                    camera_select_next=True,
                    camera_settings=camera_settings,
                )
                self.viewer.render = self.viewer.render_to_window
            else:
                self.viewer = MujocoPyRenderer(self.sim)
                self.viewer.viewer.vopt.geomgroup[0] = (
                    1 if self.render_collision_mesh else 0
                )
                self.viewer.viewer.vopt.geomgroup[1] = (
                    1 if self.render_visual_mesh else 0
                )

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
                self.renderer = DMRenderer(
                    self.sim, camera_settings=camera_settings, clear_geom_group_0=True
                )
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
                    camera_id=self.sim.model.camera_name2id(cam_name),
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
            patch_mjlib_accessors(self.get_mjlib(), self.mjpy_model, self.sim.data)

            self.renderer = DMRenderer(
                self.sim,
                clear_geom_group_0=True,
                camera_settings={},
                camera_select_next=False,
            )
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
