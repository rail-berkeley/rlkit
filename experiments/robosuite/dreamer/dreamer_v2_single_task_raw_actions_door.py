import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.arguments import get_args
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant,
)
from rlkit.torch.model_based.dreamer.experiments.raps_experiment import experiment

if __name__ == "__main__":
    args = get_args()
    num_envs = 10
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=1000,
            min_num_steps_before_training=1000,
            num_pretrain_steps=1,
            max_path_length=500,
            num_expl_steps_per_train_loop=501,
            num_trains_per_train_loop=1,
            num_train_loops_per_epoch=1,
            batch_size=50,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=100,
            num_eval_steps_per_epoch=500 * 5,
            min_num_steps_before_training=2500,
            num_pretrain_steps=100,
            max_path_length=500,
            num_expl_steps_per_train_loop=501 * num_envs,
            num_trains_per_train_loop=2000,
            num_train_loops_per_epoch=2,
            batch_size=50,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="DreamerV2",
        version="normal",
        replay_buffer_size=int(5e3),
        algorithm_kwargs=algorithm_kwargs,
        env_name="Door",
        use_raw_actions=True,
        env_suite="robosuite",
        env_kwargs=dict(
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            camera_heights=64,
            camera_widths=64,
            controller_configs={
                "type": "OSC_POSE",
                "input_max": 1,
                "input_min": -1,
                "output_max": [0.1, 0.1, 0.1, 0.5, 0.5, 0.5],
                "output_min": [-0.1, -0.1, -0.1, -0.5, -0.5, -0.5],
                "kp": 150,
                "damping_ratio": 1,
                "impedance_mode": "fixed",
                "kp_limits": [0, 300],
                "damping_ratio_limits": [0, 10],
                "position_limits": None,
                "orientation_limits": None,
                "uncouple_pos_ori": True,
                "control_delta": True,
                "interpolation": None,
                "ramp_ratio": 0.2,
            },
            horizon=500,
            control_freq=20,
            reward_shaping=False,
            reset_action_space_kwargs=dict(
                control_mode="primitives",
                action_scale=1,
                max_path_length=500,
                workspace_low=(-0.3, -0.4, 0.8),
                workspace_high=(0.0, 0.2, 1.3),
                go_to_pose_iterations=40,
                camera_settings={
                    "distance": 0.9532481338137215,
                    "lookat": [
                        -0.30494697896254724,
                        -0.4394507345054956,
                        0.9492024838769221,
                    ],
                    "azimuth": -125.859375,
                    "elevation": -29.062499923165888,
                },
            ),
            usage_kwargs=dict(
                use_dm_backend=True,
                max_path_length=500,
            ),
            image_kwargs=dict(),
        ),
        actor_kwargs=dict(
            init_std=0.0,
            num_layers=4,
            min_std=0.1,
            dist="trunc_normal",
        ),
        vf_kwargs=dict(
            num_layers=3,
        ),
        model_kwargs=dict(
            model_hidden_size=400,
            stochastic_state_size=50,
            deterministic_state_size=200,
            embedding_size=1024,
            rssm_hidden_size=200,
            reward_num_layers=2,
            pred_discount_num_layers=3,
            gru_layer_norm=True,
            std_act="sigmoid2",
        ),
        trainer_kwargs=dict(
            adam_eps=1e-5,
            discount=0.99,
            lam=0.95,
            forward_kl=False,
            free_nats=1.0,
            kl_loss_scale=0.0,
            transition_loss_scale=0.8,
            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=3e-4,
            reward_loss_scale=2.0,
            imagination_horizon=15,
            use_pred_discount=False,
            policy_gradient_loss_scale=0.0,
            actor_entropy_loss_schedule="1e-4",
            target_update_period=100,
        ),
        num_expl_envs=5 * 2,
        num_eval_envs=1,
        expl_amount=0.3,
        pass_render_kwargs=False,
        save_video=False,
    )

    search_space = {
        "env_name": [
            "Door",
        ],
        "env_kwargs.reward_shaping": [True, False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant = preprocess_variant(variant, args.debug)
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
            )
