import gym
import numpy as np
import torch
from d4rl.kitchen.kitchen_envs import (
    KitchenHingeCabinetV0,
    KitchenMicrowaveV0,
    KitchenSlideCabinetV0,
)
from gym.spaces.box import Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env


class KitchenWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env.max_steps
        self.observation_space = Box(
            0, 255, (3, self.env.imwidth, self.env.imheight), dtype=np.uint8
        )
        self.action_space = Box(
            -1, 1, (self.env.action_space.low.size,), dtype=np.float32
        )

    def reset(self):
        obs = self.env.reset()
        return obs.reshape(-1, self.env.imwidth, self.env.imheight)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs.reshape(-1, self.env.imwidth, self.env.imheight), reward, done, info


n_envs = 15
env = make_vec_env(
    KitchenMicrowaveV0,
    wrapper_class=KitchenWrapper,
    env_kwargs=dict(
        dense=False,
        image_obs=True,
        fixed_schema=False,
        action_scale=1.4,
        use_combined_action_space=True,
        proprioception=False,
        wrist_cam_concat_with_fixed_view=False,
        use_wrist_cam=False,
        normalize_proprioception_obs=True,
        use_workspace_limits=True,
        max_steps=5,
        imwidth=84,
        imheight=84,
    ),
    n_envs=n_envs,
)
torch.backends.cudnn.benchmark = True
model = PPO(
    "CnnPolicy",
    env,
    verbose=2,
    tensorboard_log="./data/kitchen_ppo_microwave/",
    n_steps=2048 // n_envs,
    batch_size=64,
    ent_coef=0.01,
    device="gpu",
)
model.learn(total_timesteps=1000000)
