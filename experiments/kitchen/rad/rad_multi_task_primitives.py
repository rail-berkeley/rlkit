import argparse
import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment


def experiment(variant):
    from rad.kitchen_train import experiment

    experiment(variant)

def dummy_exp(variant):
    for i in range(variant['num_seeds']):
        seed = random.randint(0, 100000)
        variant["seed"] = seed
        variant['run_experiment_kwargs']['mode'] ='local'
        variant['run_experiment_kwargs']['seed'] = seed
        variant['run_experiment_kwargs']['exp_prefix'] =variant['cached_exp_prefix']
        if i == variant['num_seeds']-1:
            skip_wait=False
        else:
            skip_wait=True
        run_experiment(
            experiment,
            variant=variant,
            **variant['run_experiment_kwargs'],
            skip_wait=skip_wait,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    exp_prefix = args.exp_prefix
    variant = dict(
        agent_kwargs=dict(
            discount=0.99,
            critic_lr=2e-4,
            actor_lr=2e-4,
            encoder_lr=2e-4,
            encoder_type="pixel",
            discrete_continuous_dist=False,
            data_augs="no_aug",
        ),
        num_train_steps=500000,
        frame_stack=4,
        replay_buffer_capacity=int(1e6),
        action_repeat=1,
        num_eval_episodes=5,
        init_steps=2500,
        pre_transform_image_size=64,
        image_size=64,
        batch_size=512,
        eval_freq=1000,
        log_interval=1000,
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
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=True,
                max_path_length=15,
                unflatten_images=True,
            ),
            image_kwargs=dict(),
        ),
        seed=-1,
        use_raw_actions=False,
        env_suite="kitchen",
    )

    search_space = {
        "agent_kwargs.data_augs": [
            "no_aug",
            "crop",
            "translate",
        ],
        "agent_kwargs.discrete_continuous_dist": [False],
        "env_kwargs.max_path_length": [15],
        "agent_kwargs.discount": [1-1/15, 0.95, 0.99],
        "env_name": [
            "hinge_slide_bottom_left_burner_light",
            "microwave_kettle_light_top_left_burner",
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        seed = random.randint(0, 100000)
        variant["seed"] = seed
        variant["exp_id"] = exp_id
        variant['num_seeds'] = args.num_seeds
        variant['cached_exp_prefix'] = args.exp_prefix
        variant['run_experiment_kwargs'] = dict(
            exp_prefix='test', #dump outer job to random dir
            mode=args.mode,
            use_gpu=True,
            snapshot_mode="none",
            python_cmd=subprocess.check_output("which python", shell=True).decode(
                "utf-8"
            )[:-1],
            seed=seed,
            exp_id=exp_id,
        )
        run_experiment(
            dummy_exp,
            variant=variant.copy(),
            **variant['run_experiment_kwargs']
        )
