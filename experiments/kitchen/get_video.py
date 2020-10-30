import cv2
from d4rl.kitchen.kitchen_envs import *
from hrl_exp.envs.mujoco_vec_wrappers import make_env, Async, VecEnv, DummyVecEnv
from rlkit.torch.model_based.dreamer.rollout_functions import vec_rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger
import rlkit.torch.pytorch_util as ptu

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
    elif env_class == "top_burner":
        env_class_ = KitchenTopBurnerV0
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
    path_length = 0
    while path_length < env.envs[0].max_steps:
        a, agent_info = policy.get_action(
            o,
        )
        env.step(a)
        img = env.render(mode="rgb_array", imwidth=1000, imheight=1000)
        path_length += 1
        img_array.append(img)

    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(args.file[:-10] + "final.avi", fourcc, 20.0, (1000, 1000))
    for i in range(len(img_array)):
        for _ in range(20):
            out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="path to the snapshot file")
    parser.add_argument("--H", type=int, default=3, help="Max length of rollout")
    args = parser.parse_args()

    simulate_policy(args)
