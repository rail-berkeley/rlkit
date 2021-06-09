import time

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS

from rlkit.envs.mujoco_vec_wrappers import StableBaselinesVecEnv, make_metaworld_env
from rlkit.envs.primitives_wrappers import ImageEnvMetaworld, TimeLimit

if __name__ == "__main__":
    V1_keys = [
        "reach-v1",
        "push-v1",
        "pick-place-v1",
        "door-open-v1",
        "drawer-open-v1",
        "drawer-close-v1",
        "button-press-topdown-v1",
        "peg-insert-side-v1",
        "window-open-v1",
        "window-close-v1",
        "door-close-v1",
        "reach-wall-v1",
        "pick-place-wall-v1",
        "push-wall-v1",
        "button-press-v1",
        "button-press-topdown-wall-v1",
        "button-press-wall-v1",
        "peg-unplug-side-v1",
        "disassemble-v1",
        "hammer-v1",
        "plate-slide-v1",
        "plate-slide-side-v1",
        "plate-slide-back-v1",
        "plate-slide-back-side-v1",
        "handle-press-v1",
        "handle-pull-v1",
        "handle-press-side-v1",
        "handle-pull-side-v1",
        "stick-push-v1",
        "stick-pull-v1",
        "basketball-v1",
        "soccer-v1",
        "faucet-open-v1",
        "faucet-close-v1",
        "coffee-push-v1",
        "coffee-pull-v1",
        "coffee-button-v1",
        "sweep-v1",
        "sweep-into-v1",
        "pick-out-of-hole-v1",
        "assembly-v1",
        "shelf-place-v1",
        "push-back-v1",
        "lever-pull-v1",
        "dial-turn-v1",
        "bin-picking-v1",
        "box-close-v1",
        "hand-insert-v1",
        "door-lock-v1",
        "door-unlock-v1",
    ]
    V2_keys = [
        "reach-v2",
        # "door-close-v2",
        # "sweep-into-v2",
        # "button-press-wall-v2",
        # "button-press-topdown-v2",
        # "plate-slide-v2",
        # "coffee-button-v2",
        # "handle-press-v2",
        # "window-open-v2",
        # "drawer-close-v2",
        # "handle-press-side-v2",
        # "button-press-v2",
        # "plate-slide-back-side-v2",
        # "plate-slide-side-v2",
        # "coffee-push-v2",
        # "door-unlock-v2",
        # "plate-slide-back-v2",
        # "soccer-v2",
        # "button-press-topdown-wall-v2",
        # "door-lock-v2",
        # "door-open-v2",
        # "faucet-open-v2",
        # "faucet-close-v2",
        # "handle-pull-side-v2",
        # "handle-pull-v2",
        # # "push-v2", #errors
        # # "push-wall-v2", #errors
        # "sweep-v2",
        # "window-close-v2",
        # # semi-solvable:
        # "peg-insert-side-v2",
        # "reach-wall-v2",
        # "dial-turn-v2",
        # "push-back-v2",
        # "basketball-v2",
        # "box-close-v2",
        # "coffee-pull-v2",
        # "drawer-open-v2",
        # "hand-insert-v2",
        # "lever-pull-v2",
        # "pick-place-v2",
        # "stick-pull-v2",
        # # completely unsolveable
        # "stick-push-v2",
        # # "shelf-place-v2", #errors
        # # "pick-place-wall-v2", #errors
        # "pick-out-of-hole-v2",
        # "peg-unplug-side-v2",
        # "hammer-v2",
        # "disassemble-v2",
        # "bin-picking-v2",
        # "assembly-v2",
    ]
    for env_name in V2_keys:
        print(env_name)
        # env = ImageEnvMetaworld(
        #     make_metaworld_env(
        #         env_name,
        #         dict(
        #             control_mode="primitives",
        #             use_combined_action_space=True,
        #             action_scale=0.5,
        #             max_path_length=5,
        #         ),
        #         # {}
        #     ),
        #     imwidth=64,
        #     imheight=64,
        # )
        # obs = env.reset()
        num_envs = 1
        env_fns = [
            lambda: TimeLimit(
                # ImageEnvMetaworld(
                #     make_metaworld_env(env_name, {}), imwidth=64, imheight=64
                # ),
                make_metaworld_env(env_name + "-goal-observable", {}),
                500,
            )
            for _ in range(num_envs)
        ]
        envs = StableBaselinesVecEnv(env_fns=env_fns, start_method="forkserver")
        envs.reset()
        for i in range(5 // num_envs):
            # a = np.zeros(env.action_space.low.size)
            # primitive = 'angled_x_y_grasp'
            # a[env.get_idx_from_primitive_name(primitive)] = 1
            # a[
            #         env.num_primitives
            #         + np.array(env.primitive_name_to_action_idx[primitive])
            #     ] = np.array([-np.pi / 6, -0.3, 1.4])
            a = [envs.action_space.sample() for _ in range(num_envs)]
            # print(a[-14:])
            # envs.step(
            #     a, render_every_step=False, render_mode="human"
            # )
            # print(i)
            envs.step(
                a,
            )
            # env.render()
            if i % 5 == 0:
                # if i % 150 == 0:
                # st = time.time()

                # print((time.time() - st) / (i + 1))
                envs.reset()
        # env.reset()
    # cv2.imwrite("test.png", obs.reshape(3, 64, 64).transpose(1, 2, 0))
    # plt.imshow(obs.reshape(3, 84, 84).transpose(1, 2, 0))
    # plt.show()
    # import gym

    #
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
