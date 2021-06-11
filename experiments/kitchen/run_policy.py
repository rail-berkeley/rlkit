import argparse
import subprocess
import uuid

import cv2
import torch
from d4rl.kitchen.kitchen_envs import *

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.envs.mujoco_vec_wrappers import Async, DummyVecEnv, VecEnv, make_env
from rlkit.torch.model_based.dreamer.rollout_functions import vec_rollout
from rlkit.torch.pytorch_util import set_gpu_mode

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = torch.load(args.file)
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

    env = make_env(
        env_class=env_class_,
        env_kwargs=env_params,
    )
    print("Policy loaded")
    set_gpu_mode(True)
    policy.actor.to(ptu.device)
    policy.world_model.to(ptu.device)
    i = 0
    o = env.reset()
    policy.reset()
    path_length = 0
    while path_length < env.max_steps:
        a, agent_info = policy.get_action(
            [o],
        )
        env.step(a[0], render_every_step=True)
        path_length += 1
        for im in env.img_array:
            cv2.imshow("vid", im)
            cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="path to the snapshot file")
    parser.add_argument("--H", type=int, default=3, help="Max length of rollout")
    args = parser.parse_args()

    simulate_policy(args)
