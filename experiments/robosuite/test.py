import time

import cv2
import gym
import numpy as np

from rlkit.envs.primitives_make_env import make_env

if __name__ == "__main__":
    for env_name in [
        "Lift",
        # "Stack",
        # "PickPlaceCan",
        # "NutAssemblyRound",
        # "TableWipe",
        # "DoorOpen",
    ]:
        for robot in ["Panda"]:
            env = make_env(
                "robosuite",
                env_name,
                dict(
                    robots=robot,
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    camera_heights=64,
                    camera_widths=64,
                    reward_shaping=True,
                    usage_kwargs=dict(
                        use_dm_backend=True,
                        use_raw_action_wrappers=False,
                        use_image_obs=True,
                        max_path_length=100,
                        unflatten_images=False,
                    ),
                    image_kwargs=dict(),
                ),
            )
            env.reset()
            env.reset()
            st = time.time()
            num_steps = 1000
            for i in range(1000):
                action = np.random.randn(env.robots[0].dof)  # sample random action
                obs, reward, done, info = env.step(
                    action
                )  # take action in the environment
                cv2.imwrite(
                    "test/test_{}.png".format(i), obs.reshape(64, 64, 3)[:, :, ::-1]
                )  # todo should add a wrapper to transpose the robosuite images for torch
                # env.render()  # render on display
                if done:
                    env.reset()
            print(env_name, (time.time() - st) / num_steps)

# import robosuite as suite

# create environment instance
# env = suite.make(
#     env_name="Lift", # try with other tasks like "Stack" and "Door"
#     robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
#     has_renderer=True,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
# )

# # reset the environment
# env.reset()

# for i in range(1000):
#     action = np.random.randn(env.robots[0].dof) # sample random action
#     obs, reward, done, info = env.step(action)  # take action in the environment
#     env.render()  # render on display
