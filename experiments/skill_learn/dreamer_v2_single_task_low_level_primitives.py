import argparse
import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant,
)
from rlkit.torch.model_based.dreamer.experiments.low_level_primitives_exp import (
    experiment,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=5,
            num_expl_steps_per_train_loop=5,
            min_num_steps_before_training=0,
            num_pretrain_steps=10,
            num_train_loops_per_epoch=1,
            num_trains_per_train_loop=1,
            batch_size=25,
            max_path_length=5,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=200,
            num_eval_steps_per_epoch=30,
            min_num_steps_before_training=2500,
            num_pretrain_steps=1000,
            max_path_length=5,
            batch_size=100,
            num_expl_steps_per_train_loop=30,
            num_train_loops_per_epoch=10,
            num_trains_per_train_loop=100,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="DreamerV2",
        version="normal",
        replay_buffer_size=int(1.2e4),
        algorithm_kwargs=algorithm_kwargs,
        use_raw_actions=False,
        env_suite="metaworld",
        pass_render_kwargs=True,
        env_kwargs=dict(
            control_mode="primitives",
            action_scale=1,
            max_path_length=5,
            reward_type="sparse",
            camera_settings={
                "distance": 0.38227044687537043,
                "lookat": [0.21052547, 0.32329237, 0.587819],
                "azimuth": 141.328125,
                "elevation": -53.203125160653144,
            },
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=True,
                max_path_length=5,
                unflatten_images=False,
            ),
            image_kwargs=dict(imwidth=64, imheight=64),
            collect_primitives_info=True,
            render_intermediate_obs_to_info=True,
            num_low_level_actions_per_primitive=10,
        ),
        actor_kwargs=dict(
            discrete_continuous_dist=True,
            init_std=0.0,
            num_layers=4,
            min_std=0.1,
            dist="tanh_normal_dreamer_v1",
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
            use_prior_instead_of_posterior=True,
        ),
        trainer_kwargs=dict(
            adam_eps=1e-8,
            discount=0.8,
            lam=0.95,
            forward_kl=False,
            free_nats=1.0,
            pred_discount_loss_scale=10.0,
            kl_loss_scale=0.0,
            transition_loss_scale=0.8,
            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=1e-3,
            reward_loss_scale=2.0,
            use_pred_discount=True,
            policy_gradient_loss_scale=1.0,
            actor_entropy_loss_schedule="1e-4",
            target_update_period=100,
            detach_rewards=False,
            imagination_horizon=5,
            weight_decay=0.0,
        ),
        num_expl_envs=5,
        num_eval_envs=1,
        expl_amount=0.3,
        save_video=True,
        low_level_action_dim=9,
        mlp_hidden_sizes=[512, 512],
        prioritize_fraction=0.0,
        uniform_priorities=True,
    )

    search_space = {
        "env_name": [
            "assembly-v2",
            "disassemble-v2",
            "sweep-into-v2",
            "soccer-v2",
            # "drawer-close-v2",
        ],
        "algorithm_kwargs.num_train_loops_per_epoch": [10],
        "algorithm_kwargs.num_expl_steps_per_train_loop": [30],
        "algorithm_kwargs.num_pretrain_steps": [1000],
        "algorithm_kwargs.num_trains_per_train_loop": [100],
        "algorithm_kwargs.min_num_steps_before_training": [2500],
        "algorithm_kwargs.batch_size": [100],
        "num_low_level_actions_per_primitive": [10],
        "trainer_kwargs.batch_length": [50],
        # "trainer_kwargs.binarize_rewards": [True, False],
        # "model_kwargs.reward_classifier": [True, False ],
        # "primitive_embedding": [True, False],
        # "prioritize_fraction": [0.0, 0.25, 0.5],
        # "uniform_priorities": [False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if args.debug:
            variant["algorithm_kwargs"]["num_pretrain_steps"] = 1
            variant["algorithm_kwargs"]["min_num_steps_before_training"] = 10
            variant["algorithm_kwargs"]["num_trains_per_train_loop"] = 1
        variant["replay_buffer_size"] = int(
            3e6 / (variant["num_low_level_actions_per_primitive"] * 5 + 1)
        )
        variant["trainer_kwargs"]["batch_length"] = int(
            variant["num_low_level_actions_per_primitive"] * 5
        )
        variant["env_kwargs"]["num_low_level_actions_per_primitive"] = variant[
            "num_low_level_actions_per_primitive"
        ]
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
