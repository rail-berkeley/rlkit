import cv2
import gym
import numpy as np
from gym.spaces import Box
from hrl_exp.envs.mujoco_vec_wrappers import (
    DummyVecEnv,
    StableBaselinesVecEnv,
    make_env_multiworld,
)
from metaworld import _encode_task
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

from rlkit.envs.dmc_wrappers import (
    ActionRepeat,
    ImageEnvMetaworld,
    NormalizeActions,
    TimeLimit,
)

if __name__ == "__main__":
    # env = make_env_multiworld('assembly-v2')
    # obs = env.reset()
    # for i in range(150):
    #     env.step(env.action_space.sample())
    #     env.reset()
    # cv2.imwrite("test.png", obs.reshape(3, 84, 84).transpose(1, 2, 0))

    num_envs = 2
    env_fns = [lambda: make_env_multiworld("assembly-v2") for _ in range(num_envs)]
    envs = StableBaselinesVecEnv(env_fns=env_fns, start_method="forkserver")
    envs.reset()
    d = [False] * num_envs
    import time

    st = time.time()
    for i in range(10000):
        o, r, d, _ = envs.step(
            envs.action_space.sample().reshape(1, -1).repeat(num_envs, 0)
        )
        if i % 150 == 0:
            envs.reset()
            print((time.time() - st) / (i + 1))
            print(o[0].shape)
