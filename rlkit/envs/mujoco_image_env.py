import cv2
import numpy as np
import torch
from PIL import Image
from collections.__init__ import deque

from gym import Env
from gym.spaces import Box

from rlkit.envs.wrappers import ProxyEnv


class ImageMujocoEnv(ProxyEnv, Env):
    def __init__(self,
                 wrapped_env,
                 imsize=32,
                 keep_prev=0,
                 init_camera=None,
                 camera_name=None,
                 transpose=False,
                 grayscale=False,
                 normalize=False,
                 ):
        import mujoco_py
        super().__init__(wrapped_env)

        self.imsize = imsize
        if grayscale:
            self.image_length = self.imsize * self.imsize
        else:
            self.image_length = 3 * self.imsize * self.imsize
        # This is torch format rather than PIL image
        self.image_shape = (self.imsize, self.imsize)
        # Flattened past image queue
        self.history_length = keep_prev + 1
        self.history = deque(maxlen=self.history_length)
        # init camera
        if init_camera is not None:
            sim = self._wrapped_env.sim
            viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
            init_camera(viewer.cam)
            sim.add_render_context(viewer)
        self.camera_name = camera_name  # None means default camera
        self.transpose = transpose
        self.grayscale = grayscale
        self.normalize = normalize
        self._render_local = False

        self.observation_space = Box(low=0.0,
                                     high=1.0,
                                     shape=(
                                     self.image_length * self.history_length,))

    def step(self, action):
        # image observation get returned as a flattened 1D array
        true_state, reward, done, info = super().step(action)

        observation = self._image_observation()
        self.history.append(observation)
        history = self._get_history().flatten()
        full_obs = self._get_obs(history, true_state)
        return full_obs, reward, done, info

    def reset(self, **kwargs):
        true_state = super().reset(**kwargs)
        self.history = deque(maxlen=self.history_length)

        observation = self._image_observation()
        self.history.append(observation)
        history = self._get_history().flatten()
        full_obs = self._get_obs(history, true_state)
        return full_obs

    def get_image(self):
        """TODO: this should probably consider history"""
        return self._image_observation()

    def _get_obs(self, history_flat, true_state):
        # adds extra information from true_state into to the image observation.
        # Used in ImageWithObsEnv.
        return history_flat

    def _image_observation(self):
        # returns the image as a torch format np array
        image_obs = self._wrapped_env.sim.render(width=self.imsize,
                                                 height=self.imsize,
                                                 camera_name=self.camera_name)
        if self._render_local:
            cv2.imshow('env', image_obs)
            cv2.waitKey(1)
        if self.grayscale:
            image_obs = Image.fromarray(image_obs).convert('L')
            image_obs = np.array(image_obs)
        if self.normalize:
            image_obs = image_obs / 255.0
        if self.transpose:
            image_obs = image_obs.transpose()
        return image_obs

    def _get_history(self):
        observations = list(self.history)

        obs_count = len(observations)
        for _ in range(self.history_length - obs_count):
            dummy = np.zeros(self.image_shape)
            observations.append(dummy)
        return np.c_[observations]

    def retrieve_images(self):
        # returns images in unflattened PIL format
        images = []
        for image_obs in self.history:
            pil_image = self.torch_to_pil(torch.Tensor(image_obs))
            images.append(pil_image)
        return images

    def split_obs(self, obs):
        # splits observation into image input and true observation input
        imlength = self.image_length * self.history_length
        obs_length = self.observation_space.low.size
        obs = obs.view(-1, obs_length)
        image_obs = obs.narrow(start=0,
                               length=imlength,
                               dimension=1)
        if obs_length == imlength:
            return image_obs, None

        fc_obs = obs.narrow(start=imlength,
                            length=obs.shape[1] - imlength,
                            dimension=1)
        return image_obs, fc_obs

    def enable_render(self):
        self._render_local = True


class ImageMujocoWithObsEnv(ImageMujocoEnv):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.observation_space = Box(low=0.0,
                                     high=1.0,
                                     shape=(
                                     self.image_length * self.history_length +
                                     self.wrapped_env.obs_dim,))

    def _get_obs(self, history_flat, true_state):
        return np.concatenate([history_flat,
                               true_state])