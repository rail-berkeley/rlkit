import time

import gym
from baselines.common.vec_env.subproc_chunk_vec_env import SubprocChunkVecEnv
from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS

from rlkit.envs.mujoco_vec_wrappers import StableBaselinesVecEnv, make_metaworld_env
from rlkit.envs.primitives_wrappers import ImageEnvMetaworld, TimeLimit

if __name__ == "__main__":
    test_chunk = False
    num_envs = 5
    num_chunks = 5
    start_methods = ["fork", "forkserver", "spawn"]
    # if test_chunk:
    #     envs = SubprocChunkVecEnv([make_env for idx in range(num_envs)], num_chunks)
    # else:
    #     env_fns = [make_env for _ in range(num_envs)]
    #     envs = StableBaselinesVecEnv(env_fns=env_fns, start_method="spawn")
    # print(envs.reset().shape)

    # st = time.time()
    # for i in range(10):
    #     print(i)
    #     envs.step(envs.action_space.sample().reshape(1, -1).repeat(num_envs, 0))
    # print(time.time() - st)

    start_method_time = {st: 0 for st in start_methods}
    num_steps = 1000
    for start_method in start_methods:
        env_fns = [lambda: make_metaworld_env("reach-v1") for _ in range(num_envs)]
        envs = StableBaselinesVecEnv(env_fns=env_fns, start_method=start_method)
        st = time.time()
        for i in range(num_steps):
            print(i)
            envs.step(envs.action_space.sample().reshape(1, -1).repeat(num_envs, 0))
            if i % 150 == 0:
                envs.reset
        start_method_time[start_method] = (time.time() - st) / num_steps
    print(start_method_time)

    # start_method_time = {st: 0 for st in start_methods}
    # num_steps = 1000
    # for start_method in start_methods:
    #     env_fns = [make_env_raw for _ in range(num_envs)]
    #     envs = StableBaselinesVecEnv(env_fns=env_fns, start_method=start_method)
    #     st = time.time()
    #     for i in range(num_steps):
    #         envs.step(envs.action_space.sample().reshape(1, -1).repeat(num_envs, 0))
    #     start_method_time[start_method] = (time.time() - st) / num_steps
    # print(start_method_time)
