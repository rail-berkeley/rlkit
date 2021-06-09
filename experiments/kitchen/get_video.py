import argparse
import subprocess
import uuid
from glob import glob

import cv2
import torch
from d4rl.kitchen.kitchen_envs import *

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.mujoco_vec_wrappers import DummyVecEnv, make_env
from rlkit.torch.pytorch_util import set_gpu_mode

filename = str(uuid.uuid4())


def simulate_policy(file):
    data = torch.load(file)
    policy = data["evaluation/policy"]
    env_params = data["evaluation/env_params"]
    env_class = data["evaluation/env_class"]

    if env_class == "microwave":
        env_class_ = KitchenMicrowaveV0
    elif env_class == "kettle":
        env_class_ = KitchenKettleV0
    elif env_class == "slide_cabinet":
        env_class_ = KitchenSlideCabinetV0
    elif env_class == "hinge_cabinet":
        env_class_ = KitchenHingeCabinetV0
    elif env_class == "top_left_burner":
        env_class_ = KitchenTopLeftBurnerV0
    elif env_class == "light_switch":
        env_class_ = KitchenLightSwitchV0

    eval_envs = [
        make_env(
            env_class=env_class_,
            env_kwargs=env_params,
        )
    ]
    env = DummyVecEnv(eval_envs)
    print("Policy loaded")
    set_gpu_mode(True)
    policy.actor.to(ptu.device)
    policy.world_model.to(ptu.device)
    i = 0
    img_array = []
    o = env.reset()
    policy.reset()
    path_length = 0
    while path_length < env.envs[0].max_steps:
        a, agent_info = policy.get_action(
            o,
        )
        env.step(a, render_every_step=True)
        path_length += 1
        img_array.extend(env.envs[0].img_array)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(file[:-10] + "final.avi", fourcc, 100.0, (1000, 1000))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("video saved to :", file[:-10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to the snapshot file")
    parser.add_argument("--H", type=int, default=3, help="Max length of rollout")
    args = parser.parse_args()
    for p in glob(args.path + "/*/"):
        file = p + "params.pkl"
        print(file)
        try:
            simulate_policy(file)
        except Exception as e:
            print(e)
