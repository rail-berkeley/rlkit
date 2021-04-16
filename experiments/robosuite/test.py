import time

import gym
import numpy as np

from rlkit.envs.mujoco_vec_wrappers import make_robosuite_env

if __name__ == "__main__":
    for env_name in [
        "Lift",
        "Stack",
        "PickPlaceCan",
        "NutAssemblyRound",
        "TableWipe",
        "DoorOpen",
    ]:
        for robot in ["Panda"]:
            env = make_robosuite_env(
                env_name,
                dict(
                    robots="Panda",
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    camera_heights=64,
                    camera_widths=64,
                    reward_shaping=True,
                ),
            )
            st = time.time()
            for i in range(10):
                action = np.random.randn(env.robots[0].dof)  # sample random action
                obs, reward, done, info = env.step(
                    action
                )  # take action in the environment
                # env.render()  # render on display
            print(env_name, (time.time() - st) / 1000)

    # num_envs = 10
    # env_fns = [lambda: make_robosuite_env("Lift", "Panda") for _ in range(num_envs)]
    # envs = StableBaselinesVecEnv(env_fns=env_fns, start_method="fork")
    # envs.reset()

    # st = time.time()
    # for i in range(1000 // num_envs):
    #     a = [envs.action_space.sample() for _ in range(num_envs)]
    #     envs.step(
    #         a,
    #     )
    #     # if i % 5 == 0:
    #     #     envs.reset()

    # print((time.time() - st) / 1000)
