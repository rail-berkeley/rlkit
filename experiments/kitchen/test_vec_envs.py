import time

import gym
from baselines.common.vec_env.subproc_chunk_vec_env import SubprocChunkVecEnv
from d4rl.kitchen.kitchen_envs import (
    KitchenHingeCabinetV0,
    KitchenHingeSlideBottomLeftBurnerLightV0,
    KitchenKettleV0,
    KitchenLightSwitchV0,
    KitchenMicrowaveKettleLightTopLeftBurnerV0,
    KitchenMicrowaveV0,
    KitchenSlideCabinetV0,
    KitchenTopLeftBurnerV0,
)
from fromrlkit.envs.mujoco_vec_wrappers import (
    DummyVecEnv,
    StableBaselinesVecEnv,
    make_env,
)


def make_env():
    env_class_ = KitchenSlideCabinetV0
    env_pre = env_class_(
        dense=False,
        image_obs=True,
        fixed_schema=False,
        action_scale=1.4,
        use_combined_action_space=True,
        proprioception=False,
        wrist_cam_concat_with_fixed_view=False,
        use_wrist_cam=False,
        normalize_proprioception_obs=True,
        use_workspace_limits=True,
        imwidth=64,
        imheight=64,
    )
    return env_pre


def make_env_raw():
    env_class_ = KitchenSlideCabinetV0
    env_pre = env_class_(
        dense=False,
        image_obs=True,
        fixed_schema=False,
        action_scale=1.4,
        use_combined_action_space=True,
        proprioception=False,
        wrist_cam_concat_with_fixed_view=False,
        use_wrist_cam=False,
        normalize_proprioception_obs=True,
        use_workspace_limits=True,
        max_steps=1000,
        control_mode="joint_velocity",
        imwidth=64,
        imheight=64,
    )
    return env_pre


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
        env_fns = [make_env for _ in range(num_envs)]
        envs = StableBaselinesVecEnv(env_fns=env_fns, start_method=start_method)
        st = time.time()
        for i in range(num_steps):
            print(i)
            envs.step(envs.action_space.sample().reshape(1, -1).repeat(num_envs, 0))
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
