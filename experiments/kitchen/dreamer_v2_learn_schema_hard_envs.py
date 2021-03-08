import argparse
import random

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant,
)
from rlkit.torch.model_based.dreamer.experiments.kitchen_dreamer import experiment

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
            num_eval_steps_per_epoch=10,
            num_expl_steps_per_train_loop=50,
            min_num_steps_before_training=10,
            num_pretrain_steps=10,
            num_train_loops_per_epoch=1,
            num_trains_per_train_loop=10,
            batch_size=119,
            max_path_length=20,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=1000,
            min_num_steps_before_training=2500,
            num_pretrain_steps=100,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="DreamerV2",
        version="normal",
        algorithm_kwargs=algorithm_kwargs,
        env_class="hinge_cabinet",
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
        ),
        trainer_kwargs=dict(
            use_amp=True,
            opt_level="O1",
            optimizer_class="apex_adam",
            adam_eps=1e-5,
            discount=0.99,
            lam=0.95,
            forward_kl=False,
            free_nats=1.0,
            pred_discount_loss_scale=10.0,
            kl_loss_scale=0.0,
            transition_loss_scale=0.8,
            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=3e-4,
            reward_loss_scale=2.0,
            use_pred_discount=True,
            policy_gradient_loss_scale=1.0,
            actor_entropy_loss_schedule="1e-4",
            target_update_period=100,
            detach_rewards=False,
        ),
        num_expl_envs=5,
        num_eval_envs=1,
        expl_amount=0.3,
    )

    search_space = {
        "env_class": [
            "microwave_kettle_light_top_left_burner",
            "hinge_slide_bottom_left_burner_light",
        ],
        "trainer_kwargs.discount": [0.99, 0.95],
        "trainer_kwargs.actor_entropy_loss_schedule": [
            "1e-4",
        ],
        "max_steps": [15],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        max_steps = variant["max_steps"]
        replay_buffer_size = 2500000 // max_steps

        num_eval_steps_per_epoch = 5 * (max_steps + 1)
        max_path_length = max_steps
        batch_size = 2500 // (max_steps + 1)
        num_expl_steps_per_train_loop = 5 * (max_steps + 1)
        num_train_loops_per_epoch = 1000 // (5 * max_steps)
        num_trains_per_train_loop = 400 // (num_train_loops_per_epoch)
        variant["algorithm_kwargs"][
            "num_eval_steps_per_epoch"
        ] = num_eval_steps_per_epoch
        variant["algorithm_kwargs"]["max_path_length"] = max_steps
        variant["algorithm_kwargs"]["batch_size"] = batch_size
        variant["algorithm_kwargs"][
            "num_expl_steps_per_train_loop"
        ] = num_expl_steps_per_train_loop
        variant["algorithm_kwargs"][
            "num_train_loops_per_epoch"
        ] = num_train_loops_per_epoch
        variant["algorithm_kwargs"][
            "num_trains_per_train_loop"
        ] = num_trains_per_train_loop
        variant["replay_buffer_size"] = replay_buffer_size
        variant["env_kwargs"]["max_steps"] = max_steps
        variant["trainer_kwargs"]["imagination_horizon"] = min(max_steps, 15)
        if variant["trainer_kwargs"]["discount"] != 0.99:
            variant["trainer_kwargs"]["discount"] = 1 - 1 / max_steps
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
                python_cmd="~/miniconda3/envs/hrl-exp-env/bin/python",
                seed=seed,
                exp_id=exp_id,
            )
