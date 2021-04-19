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
    V2_keys = [
        "reach-v2",
            "door-close-v2",
            "sweep-into-v2",
            "button-press-wall-v2",
            "button-press-topdown-v2",
            "plate-slide-v2",
            "coffee-button-v2",
            "handle-press-v2",
            "window-open-v2",
            "drawer-close-v2",
            "handle-press-side-v2",
            "button-press-v2",
            "plate-slide-back-side-v2",
            "plate-slide-side-v2",
            "coffee-push-v2",
            "door-unlock-v2",
            "plate-slide-back-v2",
            "soccer-v2",
            "button-press-topdown-wall-v2",
            "door-lock-v2",
            "door-open-v2",
            "faucet-open-v2",
            "faucet-close-v2",
            "handle-pull-side-v2",
            "handle-pull-v2",
            "push-v2", #errors
            "push-wall-v2", #errors
            "sweep-v2",
            "window-close-v2",
            # semi-solvable:
            "peg-insert-side-v2",
            "reach-wall-v2",
            "dial-turn-v2",
            "push-back-v2",
            "basketball-v2",
            "box-close-v2",
            "coffee-pull-v2",
            "drawer-open-v2",
            "hand-insert-v2",
            "lever-pull-v2",
            "pick-place-v2",
            "stick-pull-v2",
            # completely unsolveable
            "stick-push-v2",
            "shelf-place-v2", #errors
            "pick-place-wall-v2", #errors
            "pick-out-of-hole-v2",
            "peg-unplug-side-v2",
            "hammer-v2",
            "disassemble-v2",
            "bin-picking-v2",
            "assembly-v2",
    ]
    for env_name in V2_keys:
        print(env_name)
        num_envs = 1
        env = make_metaworld_env(env_name + "-goal-observable", {}, use_dm_backend=True)
        env = ImageEnvMetaworld(env, imwidth=400, imheight=400)
        env.reset()
        for i in range(1):
            # primitive = 'angled_x_y_grasp'
            # a[env.get_idx_from_primitive_name(primitive)] = 1
            # a[
            #         env.num_primitives
            #         + np.array(env.primitive_name_to_action_idx[primitive])
            #     ] = np.array([-np.pi / 6, -0.3, 1.4])
            env.step(
                env.action_space.sample(),
            )
            # env.render()
            env.save_image()
            if i % 500 == 0:
                # if i % 150 == 0:
                # st = time.time()

                # print((time.time() - st) / (i + 1))
                env.reset()
