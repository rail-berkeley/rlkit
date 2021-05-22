import argparse
import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment


def experiment(variant):
    from spirl.rl.train import experiment

    experiment(variant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    exp_prefix = args.exp_prefix
    variant = dict(
        env_kwargs=dict(
            dense=False,
            image_obs=False,
            fixed_schema=False,
            action_scale=1,
            use_combined_action_space=True,
            proprioception=False,
            wrist_cam_concat_with_fixed_view=False,
            use_wrist_cam=False,
            normalize_proprioception_obs=True,
            use_workspace_limits=True,
            max_path_length=280,
            control_mode="joint_velocity",
            frame_skip=40,
            imwidth=64,
            imheight=64,
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=False,
                max_path_length=280,
                unflatten_images=False,
            ),
            image_kwargs=dict(),
        ),
        env_suite="kitchen",
        path="/home/mdalal/research/spirl/spirl/configs/hrl/kitchen/vision_cl_1/",
        env_name="kettle",
    )

    search_space = {
        "env_name": [
            "microwave",
            "kettle",
            "slide_cabinet",
            "top_left_burner",
            # "hinge_cabinet",
            # "light_switch",
            # "microwave_kettle_light_top_left_burner",
            # "hinge_slide_bottom_left_burner_light",
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(args.num_seeds):
            seed = random.randint(0, 100000)
            variant["seed"] = seed
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
                skip_wait=True,
            )
