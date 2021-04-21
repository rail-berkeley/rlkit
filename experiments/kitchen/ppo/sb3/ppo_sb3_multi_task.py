import argparse
import os
import random
import subprocess

import numpy as np
import torch

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment


def experiment(variant):
    import gym
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
    from gym.spaces.box import Box
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env

    from rlkit.core import logger

    class KitchenWrapper(gym.Wrapper):
        def __init__(self, env):
            gym.Wrapper.__init__(self, env)
            self._max_episode_steps = env.max_steps
            self.observation_space = Box(
                0, 255, (3, self.env.imwidth, self.env.imheight), dtype=np.uint8
            )
            self.action_space = Box(
                -1, 1, (self.env.action_space.low.size,), dtype=np.float32
            )

        def reset(self):
            obs = self.env.reset()
            return obs.reshape(-1, self.env.imwidth, self.env.imheight)

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            return (
                obs.reshape(-1, self.env.imwidth, self.env.imheight),
                reward,
                done,
                info,
            )

    env_class = variant["env_class"]
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
    elif env_class == "microwave_kettle_light_top_left_burner":
        env_class_ = KitchenMicrowaveKettleLightTopLeftBurnerV0
    elif env_class == "hinge_slide_bottom_left_burner_light":
        env_class_ = KitchenHingeSlideBottomLeftBurnerLightV0
    else:
        raise EnvironmentError("invalid env provided")
    n_envs = variant["n_envs"]
    env = make_vec_env(
        env_class_,
        wrapper_class=KitchenWrapper,
        n_envs=n_envs,
        env_kwargs=variant["env_kwargs"],
    )
    model = PPO(
        "CnnPolicy",
        env,
        tensorboard_log=logger.get_snapshot_dir() + "/" + env_class,
        n_steps=2048 // n_envs,
        device="cuda",
        **variant["algorithm_kwargs"]
    )

    def train():
        torch.backends.cudnn.enabled = True
        model.learn(total_timesteps=variant["total_timesteps"])

    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    if args.debug:
        algorithm_kwargs = dict()
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            ent_coef=0.01,
            learning_rate=3e-4,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm_kwargs=algorithm_kwargs,
        n_envs=12,
        total_timesteps=100000,
        env_kwargs=dict(
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
            max_steps=10,
            imwidth=84,
            imheight=84,
        ),
    )

    search_space = {
        "algorithm_kwargs.gamma": [0.99, 0.95],
        "algorithm_kwargs.ent_coef": [0.01],
        "total_timesteps": [500000],
        "env_class": [
            "microwave_kettle_light_top_left_burner",
            "hinge_slide_bottom_left_burner_light",
        ],
        "env_kwargs.max_steps": [10, 15, 20],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(args.num_seeds):
            seed = random.randint(0, 100000)
            variant["seed"] = seed
            variant["algorithm_kwargs"]["seed"] = seed
            variant["exp_id"] = exp_id
            run_experiment(
                experiment,
                exp_prefix=args.exp_prefix,
                mode=args.mode,
                variant=variant,
                use_gpu=True,
                snapshot_mode="none",
                python_cmd=subprocess.check_output("which python", shell=True).decode(
                    "utf-8"
                )[:-1],
                seed=seed,
                exp_id=exp_id,
            )
