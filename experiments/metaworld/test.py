import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from rlkit.envs.mujoco_vec_wrappers import StableBaselinesVecEnv, make_metaworld_env
from rlkit.envs.primitives_wrappers import ImageEnvMetaworld, TimeLimit

if __name__ == "__main__":
    # env = make_metaworld_env("assembly-v2", dict(control_mode='primitives', use_combined_action_space=True, action_scale=1.4))
    env = ImageEnvMetaworld(
        make_metaworld_env("assembly-v2", {}), imwidth=84, imheight=84
    )
    obs = env.reset()
    for i in range(10000):
        # a = np.zeros(env.action_space.low.size)
        # primitive = 'angled_x_y_grasp'
        # a[env.get_idx_from_primitive_name(primitive)] = 1
        # a[
        #         env.num_primitives
        #         + np.array(env.primitive_name_to_action_idx[primitive])
        #     ] = np.array([-np.pi / 6, -0.3, 1.4])
        env.step(env.action_space.sample())
        env.render(mode='human')
        if i % 150 == 0:
            env.reset()
        # env.reset()
    # cv2.imwrite("test.png", obs.reshape(3, 64, 64).transpose(1, 2, 0))
        plt.imshow(obs.reshape(3, 84, 84).transpose(1, 2, 0))
        plt.show()
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
