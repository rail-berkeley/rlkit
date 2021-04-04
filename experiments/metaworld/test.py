import cv2
import gym
import numpy as np
from gym.spaces import Box

from rlkit.envs.mujoco_vec_wrappers import StableBaselinesVecEnv, make_metaworld_env
from rlkit.envs.primitives_wrappers import ImageEnvMetaworld, TimeLimit

if __name__ == "__main__":
    env = TimeLimit(
        ImageEnvMetaworld(
            make_metaworld_env("assembly-v2", {}), imwidth=64, imheight=64
        ),
        150,
    )
    obs = env.reset()
    for i in range(150):
        env.step(env.action_space.sample())
        env.reset()
    # cv2.imwrite("test.png", obs.reshape(3, 84, 84).transpose(1, 2, 0))
    # import gym

    # gym.logger.set_level(40)
    # num_envs = 15
    # performance = {}
    # for num_envs in range(33, 34):
    #     env_fns = [
    #         lambda: TimeLimit(
    #             ImageEnvMetaworld(
    #                 make_metaworld_env("assembly-v2", {}), imwidth=64, imheight=64
    #             ),
    #             150,
    #         )
    #         for _ in range(num_envs)
    #     ]
    #     envs = StableBaselinesVecEnv(env_fns=env_fns, start_method="forkserver")
    #     envs.reset()
    #     d = [False] * num_envs
    #     import time

    #     st = time.time()
    #     for i in range(10000):
    #         o, r, d, _ = envs.step(
    #             envs.action_space.sample().reshape(1, -1).repeat(num_envs, 0)
    #         )
    #         if i % 150 == 0:
    #             envs.reset()
    #     performance[num_envs] = (time.time() - st) / 10000 / num_envs
    # print(performance)
