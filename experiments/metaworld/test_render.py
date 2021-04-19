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
    ]
    for env_name in V2_keys:
        print(env_name)
        num_envs = 1
        env = make_metaworld_env(
            env_name,
            dict(
                control_mode="primitives",
                use_combined_action_space=True,
                action_scale=1,
                max_path_length=10,
                use_image_obs=False,
                reward_scale=1,  # let VecNormalize handle the reward scales
                use_dm_backend=True,
                remove_rotation_primitives=True,
            ),
            use_dm_backend=True,
        )
        o = env.reset()
        print(o[-3:])
        for i in range(1000):
            a = env.action_space.sample()
            a = np.zeros_like(a)
            primitive = "move_delta_ee_pose"
            a[env.get_idx_from_primitive_name(primitive)] = 1
            delta = o[-3:] - o[:3]
            a[
                env.num_primitives
                + np.array(env.primitive_name_to_action_idx[primitive])
            ] = delta
            o, r, d, info = env.step(
                a,
            )
            print(delta)
            # print(info['near_object'], info['success'])
            env.render()
            if i % 10 == 0:
                env.reset()
