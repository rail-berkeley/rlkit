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
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_assembly_peg_v2 import (
    SawyerNutAssemblyEnvV2,
)


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

    if not hasattr(model, "get_joint_qpos_addr"):

        def get_joint_qpos_addr(name):
            joint_id = model.joint_name2id(name)
            joint_type = model.jnt_type[joint_id]
            joint_addr = model.jnt_qposadr[joint_id]
            # TODO: remove hardcoded joint ids (find where mjtJoint is)
            if joint_type == 0:
                ndim = 7
            elif joint_type == 1:
                ndim = 4
            else:
                assert joint_type in (2, 3)
                ndim = 1

            if ndim == 1:
                return joint_addr
            else:
                return (joint_addr, joint_addr + ndim)

        model.get_joint_qpos_addr = lambda name: get_joint_qpos_addr(name)

    if not hasattr(model, "get_joint_qvel_addr"):

        def get_joint_qvel_addr(name):
            joint_id = model.joint_name2id(name)
            joint_type = model.jnt_type[joint_id]
            joint_addr = model.jnt_dofadr[joint_id]
            if joint_type == 0:
                ndim = 6
            elif joint_type == 1:
                ndim = 3
            else:
                assert joint_type in (3, 2)
                ndim = 1

            if ndim == 1:
                return joint_addr
            else:
                return (joint_addr, joint_addr + ndim)

        model.get_joint_qvel_addr = lambda name: get_joint_qvel_addr(name)

    if not hasattr(data, "body_xpos"):
        data.body_xpos = data.xpos

    if not hasattr(data, "body_xquat"):
        data.body_xquat = data.xquat

    if not hasattr(data, "body_xmat"):
        data.body_xmat = data.xmat

    if not hasattr(data, "get_body_xpos"):
        data.get_body_xpos = lambda name: data.body_xpos[model.body_name2id(name)]

    if not hasattr(data, "get_body_xquat"):
        data.get_body_xquat = lambda name: data.body_xquat[model.body_name2id(name)]

    if not hasattr(data, "get_body_xmat"):
        data.get_body_xmat = lambda name: data.xmat[model.body_name2id(name)].reshape(
            (3, 3)
        )

    if not hasattr(data, "get_geom_xpos"):
        data.get_geom_xpos = lambda name: data.geom_xpos[model.geom_name2id(name)]

    if not hasattr(data, "get_geom_xquat"):
        data.get_geom_xquat = lambda name: data.geom_xquat[model.geom_name2id(name)]

    if not hasattr(data, "get_joint_qpos"):

        def get_joint_qpos(name):
            addr = model.get_joint_qpos_addr(name)
            if isinstance(addr, (int, np.int32, np.int64)):
                return data.qpos[addr]
            else:
                start_i, end_i = addr
                return data.qpos[start_i:end_i]

        data.get_joint_qpos = lambda name: get_joint_qpos(name)

    if not hasattr(data, "set_joint_qpos"):

        def set_joint_qpos(name, value):
            addr = model.get_joint_qpos_addr(name)
            if isinstance(addr, (int, np.int32, np.int64)):
                data.qpos[addr] = value
            else:
                start_i, end_i = addr
                value = np.array(value)
                assert value.shape == (
                    end_i - start_i,
                ), "Value has incorrect shape %s: %s" % (name, value)
                data.qpos[start_i:end_i] = value

        data.set_joint_qpos = lambda name, value: set_joint_qpos(name, value)

    if not hasattr(data, "get_site_xmat"):
        data.get_site_xmat = lambda name: data.site_xmat[
            model.site_name2id(name)
        ].reshape((3, 3))

    if not hasattr(data, "get_geom_xmat"):
        data.get_geom_xmat = lambda name: data.geom_xmat[
            model.geom_name2id(name)
        ].reshape((3, 3))

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
            (3, model.nv)
        )

    if not hasattr(data, "get_site_jacr"):
        data.get_site_jacr = lambda name: site_jacr()[model.site_name2id(name)].reshape(
            (3, model.nv)
        )

    def body_jacp():
        jacps = np.zeros((model.nbody, 3 * model.nv))
        for i, jacp in enumerate(jacps):
            jacp_view = jacp
            mjlib.mj_jacBody(model.ptr, data.ptr, jacp_view, None, i)
        return jacps

    def body_xvelp():
        jacp = body_jacp().reshape((model.nbody, 3, model.nv))
        xvelp = np.dot(jacp, data.qvel)
        return xvelp

    def body_jacr():
        jacrs = np.zeros((model.nbody, 3 * model.nv))
        for i, jacr in enumerate(jacrs):
            jacr_view = jacr
            mjlib.mj_jacBody(model.ptr, data.ptr, None, jacr_view, i)
        return jacrs

    def body_xvelr():
        jacp = body_jacr().reshape((model.nbody, 3, model.nv))
        xvelp = np.dot(jacp, data.qvel)
        return xvelp

    if not hasattr(data, "body_xvelp"):
        data.body_xvelp = body_xvelp()

    if not hasattr(data, "body_xvelr"):
        data.body_xvelr = body_xvelr()

    if not hasattr(data, "get_body_jacp"):
        data.get_body_jacp = lambda name: body_jacp()[model.body_name2id(name)].reshape(
            (3, model.nv)
        )

    if not hasattr(data, "get_body_jacr"):
        data.get_body_jacr = lambda name: body_jacr()[model.body_name2id(name)].reshape(
            (3, model.nv)
        )


class DMControlBackendMetaworldMujocoEnv(MujocoEnv):
    def __init__(self, model_path, frame_skip, rgb_array_res=(640, 480)):
        if not path.exists(model_path):
            raise IOError("File %s does not exist" % model_path)
        if self.control_mode == "vices":
            if model_path == (
                "/home/mdalal/research/metaworld/metaworld/envs/assets_v2/sawyer_xyz/sawyer_basketball.xml"
            ):
                model_path = "/home/mdalal/research/metaworld/metaworld/envs/assets_v2/sawyer_xyz/sawyer_basketball_torque.xml"
                self.reset_qpos = np.array(
                    [
                        0.00000000e00,
                        6.00000000e-01,
                        2.98721632e-02,
                        1.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        1.88500731e00,
                        -5.88898933e-01,
                        -2.50919766e-02,
                        6.95071420e-01,
                        2.99999993e-02,
                        1.02969815e00,
                        2.31183042e00,
                        -1.71909704e-04,
                        1.71743732e-04,
                    ]
                )
            elif model_path == (
                "/home/mdalal/research/metaworld/metaworld/envs/assets_v2/sawyer_xyz/sawyer_drawer.xml"
            ):
                model_path = "/home/mdalal/research/metaworld/metaworld/envs/assets_v2/sawyer_xyz/sawyer_drawer_torque.xml"
                self.reset_qpos = [
                    1.88500731e00,
                    -5.88898933e-01,
                    -9.64183847e-01,
                    1.64749509e00,
                    9.28632075e-01,
                    1.02969815e00,
                    2.31183042e00,
                    -1.71909704e-04,
                    1.71743732e-04,
                    -1.50000000e-01,
                ]
            elif model_path == (
                "/home/mdalal/research/metaworld/metaworld/envs/assets_v2/sawyer_xyz/sawyer_soccer.xml"
            ):
                model_path = "/home/mdalal/research/metaworld/metaworld/envs/assets_v2/sawyer_xyz/sawyer_soccer_torque.xml"
                self.reset_qpos = [
                    1.88500734e00,
                    -5.88898932e-01,
                    -9.64183883e-01,
                    1.64749516e00,
                    9.28632122e-01,
                    1.02969813e00,
                    2.31183036e00,
                    -1.71909704e-04,
                    1.71743732e-04,
                    -8.83832789e-02,
                    6.86617607e-01,
                    3.00000000e-02,
                    1.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ]
            elif model_path == (
                "/home/mdalal/research/metaworld/metaworld/envs/assets_v2/sawyer_xyz/sawyer_table_with_hole.xml"
            ):
                model_path = "/home/mdalal/research/metaworld/metaworld/envs/assets_v2/sawyer_xyz/sawyer_table_with_hole_torque.xml"
                self.reset_qpos = [
                    1.88500734e00,
                    -5.88898932e-01,
                    -9.64183883e-01,
                    1.64749516e00,
                    9.28632122e-01,
                    1.02969813e00,
                    2.31183036e00,
                    -1.71909704e-04,
                    1.71743732e-04,
                    -8.83832789e-02,
                    6.86617607e-01,
                    6.99354863e-02,
                    1.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ]
            elif model_path == (
                "/home/mdalal/research/metaworld/metaworld/envs/assets_v2/sawyer_xyz/sawyer_assembly_peg.xml"
            ):
                model_path = "/home/mdalal/research/metaworld/metaworld/envs/assets_v2/sawyer_xyz/sawyer_assembly_peg_torque.xml"
                if type(self) == SawyerNutAssemblyEnvV2:
                    self.reset_qpos = [
                        1.88500731e00,
                        -5.88898933e-01,
                        -9.64183847e-01,
                        1.64749509e00,
                        9.28632075e-01,
                        1.02969815e00,
                        2.31183042e00,
                        -1.71909704e-04,
                        1.71743732e-04,
                        0.00000000e00,
                        6.00000024e-01,
                        1.99999996e-02,
                        7.07106763e-01,
                        1.36367940e-04,
                        1.60057621e-04,
                        7.07106768e-01,
                    ]
                else:
                    self.reset_qpos = [
                        2.16526293e00,
                        -5.57054189e-01,
                        -1.22832464e00,
                        2.17377367e00,
                        1.24233511e00,
                        1.00706272e00,
                        1.91061865e00,
                        -1.71855687e-04,
                        1.71795647e-04,
                        6.25459890e-02,
                        7.42607147e-01,
                        2.50073207e-02,
                        7.07106763e-01,
                        1.36367940e-04,
                        1.60057621e-04,
                        7.07106768e-01,
                    ]
            else:
                raise ValueError("no model found")

        self.frame_skip = frame_skip
        self._use_dm_backend = True

        # old zoomed out view
        # camera_settings = {
        #     "distance": 1.878359835328275,
        #     "lookat": [0.16854934, 0.27084485, 0.23161897],
        #     "azimuth": 141.328125,
        #     "elevation": -53.203125160653144,
        # }

        # intermediate view
        # camera_settings = {
        #     "distance":0.8304056576521722,
        #     "lookat":[0.21052547, 0.32329237, 0.587819 ],
        #     "azimuth": 141.328125,
        #     "elevation": -53.203125160653144,
        # }

        # super zoomed in - working
        # camera_settings = {
        #     "distance": 0.38227044687537043,
        #     "lookat": [0.21052547, 0.32329237, 0.587819],
        #     "azimuth": 141.328125,
        #     "elevation": -53.203125160653144,
        # }

        # side view - semi-close up
        # camera_settings = {
        #     "distance":0.513599996134662,
        #     "lookat":[0.28850459, 0.56757972, 0.54530015],
        #     "azimuth": 178.9453125,
        #     "elevation": -60.00000040512532,
        # }

        # side view super zoomed in
        camera_settings = {
            "distance": 0.31785821791481395,
            "lookat": [0.28850459, 0.56757972, 0.54530015],
            "azimuth": 178.59375,
            "elevation": -60.46875041909516,
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
        imwidth=64,
        imheight=64,
    ):
        if mode == "human":
            self.renderer.render_to_window()
        elif mode == "rgb_array":
            return self.renderer.render_offscreen(
                imwidth,
                imheight,
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


class DMControlBackendRobosuiteEnv(robosuite.environments.base.MujocoEnv):
    def _reset_internal(self):
        if self._use_dm_backend:
            if self.has_renderer and self.viewer is None:
                self.viewer = DMRenderer(
                    self.dm_sim,
                    clear_geom_group_0=True,
                    camera_select_next=False,
                    camera_settings=self.camera_settings,
                    mjpy_sim=self.sim,
                )
                self.viewer.render = self.viewer.render_to_window
            elif self.has_offscreen_renderer:
                self.renderer = DMRenderer(
                    self.dm_sim,
                    camera_settings=self.camera_settings,
                    clear_geom_group_0=True,
                    mjpy_sim=self.sim,
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
        else:
            super()._reset_internal()

    def _initialize_sim(self, xml_string=None):
        """
        Creates a MjSim object and stores it in self.sim. If @xml_string is specified, the MjSim object will be created
        from the specified xml_string. Else, it will pull from self.model to instantiate the simulation

        Args:
            xml_string (str): If specified, creates MjSim object from this filepath
        """
        super()._initialize_sim(xml_string)
        if self._use_dm_backend:
            if not hasattr(self, "camera_settings"):
                self.camera_settings = {}
            with io.StringIO() as string:
                string.write(ET.tostring(self.model.root, encoding="unicode"))
                st = string.getvalue()

            dm_mujoco = module.get_dm_mujoco()
            self.dm_sim = dm_mujoco.Physics.from_xml_string(st)
            patch_mjlib_accessors(
                module.get_dm_mujoco().wrapper.mjbindings.mjlib,
                self.dm_sim.model,
                self.dm_sim.data,
            )
            self.renderer = DMRenderer(
                self.dm_sim,
                clear_geom_group_0=True,
                camera_settings=self.camera_settings,
                camera_select_next=False,
                mjpy_sim=self.sim,
            )

    def get_mjlib(self):
        """Returns an object that exposes the low-level MuJoCo API."""
        return module.get_dm_mujoco().wrapper.mjbindings.mjlib
