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
    eval_env = DummyVecEnv(eval_envs)
    print("Policy loaded")
    set_gpu_mode(True)
    policy.actor.to(ptu.device)
    policy.world_model.to(ptu.device)
    i = 0
    while True:
        path = vec_rollout(
            eval_env,
            policy,
            max_path_length=eval_env.envs[0].max_steps,
            render=True,
        )
        if hasattr(eval_env, "log_diagnostics"):
            eval_env.log_diagnostics([path])
        logger.dump_tabular()
        i += 1
        print("path: ", i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="path to the snapshot file")
    parser.add_argument("--H", type=int, default=3, help="Max length of rollout")
    args = parser.parse_args()

    simulate_policy(args)
