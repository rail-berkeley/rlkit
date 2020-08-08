import numpy as np
from gym.spaces import Box

from rlkit.envs.wrappers.image_mujoco_env import ImageMujocoEnv


class ImageMujocoWithObsEnv(ImageMujocoEnv):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(self.image_length * self.history_length
                   + self.wrapped_env.obs_dim,))

    def _get_obs(self, history_flat, true_state):
        return np.concatenate([history_flat, true_state])
