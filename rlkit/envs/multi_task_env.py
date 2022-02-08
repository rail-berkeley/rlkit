import gc

import gym
import numpy as np

import rlkit.envs.primitives_make_env as primitives_make_env


class MultiTaskEnv(gym.Env):
    def __init__(self, env_suite, env_names, env_kwargs):
        """
        Multi-task environment which on reset, samples a new env.
        Observations from each env are concatenated with a one-hot vector giving the task index.
        Note: you must collect at least one trajectory from each environment per epoch otherwise
        the logging code will break due to missing keys.
        """
        self.env_kwargs = env_kwargs
        self.env_names = env_names
        self.env_suite = env_suite
        self.num_resets = 0
        self.num_multitask_envs = len(env_names)
        self.reset()
        # assume action space is constant across envs
        old_obs_space = self.env.observation_space
        self.observation_space = gym.spaces.Box(
            np.concatenate((old_obs_space.low, np.zeros(self.num_multitask_envs))),
            np.concatenate((old_obs_space.high, np.ones(self.num_multitask_envs))),
        )
        self.action_space = self.env.action_space

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
        obs = np.concatenate((obs, self.get_one_hot(self.idx)))
        new_info = {}
        for k, v in info.items():
            new_info[self.env_names[self.idx] + "/" + k] = v
            new_info[k] = v
        return obs, reward, done, new_info

    def reset(self):
        if hasattr(self, "env"):
            del self.env
            gc.collect()
        self.idx = self.num_resets % self.num_multitask_envs
        env_name = self.env_names[self.idx]
        self.env = primitives_make_env.make_env(
            self.env_suite, env_name, self.env_kwargs
        )
        o = self.env.reset()
        self.num_resets += 1
        o = np.concatenate((o, self.get_one_hot(self.idx)))
        return o

    def get_one_hot(self, idx):
        one_hot = np.zeros(self.num_multitask_envs)
        one_hot[idx] = 1
        return one_hot

    def __getattr__(self, name):
        if name != "env":
            return getattr(self.env, name)
        else:
            raise AttributeError("")


if __name__ == "__main__":
    env_kwargs = dict(
        control_mode="primitives",
        action_scale=1,
        max_path_length=5,
        reward_type="sparse",
        camera_settings={
            "distance": 0.38227044687537043,
            "lookat": [0.21052547, 0.32329237, 0.587819],
            "azimuth": 141.328125,
            "elevation": -53.203125160653144,
        },
        usage_kwargs=dict(
            use_dm_backend=True,
            use_raw_action_wrappers=False,
            use_image_obs=True,
            max_path_length=5,
            unflatten_images=False,
        ),
    )
    env_names = [
        "assembly-v2",
        "disassemble-v2",
        "peg-unplug-side-v2",
        "soccer-v2",
        "sweep-into-v2",
        "drawer-close-v2",
    ]
    env_suite = "metaworld"

    env = MultiTaskEnv(env_suite, env_names, env_kwargs)

    o = env.reset()
    print(o.shape)
    print(env.observation_space.shape)
    print(env.action_space.shape)
    i = 0
    while True:
        env.step(env.action_space.sample())

        print(i)
        if i % 5 == 0:
            env.reset()
        i += 1
