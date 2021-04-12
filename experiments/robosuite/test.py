import time

import gym
import numpy as np
import robosuite as suite

from rlkit.envs.mujoco_vec_wrappers import StableBaselinesVecEnv


class RobosuiteWrapper:
    def __init__(self, env):
        self.env = env
        image_shape = env.observation_spec()["agentview_image"].shape
        self.observation_space = gym.spaces.Box(0, 255, image_shape, dtype=np.uint8)
        self.action_space = gym.spaces.Box(
            env.action_spec[0], env.action_spec[1], dtype=np.float32
        )

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_robosuite_env(env_name, robots, imwidth=64, imheight=64):
    # create environment instance
    gym.logger.setLevel(40)
    env = RobosuiteWrapper(
        suite.make(
            env_name=env_name,  # try with other tasks like "Stack" and "Door"
            robots=robots,  # try with other robots like "Sawyer" and "Jaco"
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_heights=imheight,
            camera_widths=imwidth,
            reward_shaping=True,
        )
    )
    return env


if __name__ == "__main__":
    # env = make_robosuite_env('Lift', 'Panda')
    # st = time.time()
    # for i in range(1000):
    #     action = np.random.randn(env.robots[0].dof) # sample random action
    #     obs, reward, done, info = env.step(action)  # take action in the environment
    #     # env.render()  # render on display
    # print((time.time()-st)/1000)

    num_envs = 10
    env_fns = [lambda: make_robosuite_env("Lift", "Panda") for _ in range(num_envs)]
    envs = StableBaselinesVecEnv(env_fns=env_fns, start_method="fork")
    envs.reset()

    st = time.time()
    for i in range(1000 // num_envs):
        a = [envs.action_space.sample() for _ in range(num_envs)]
        envs.step(
            a,
        )
        # if i % 5 == 0:
        #     envs.reset()

    print((time.time() - st) / 1000)
