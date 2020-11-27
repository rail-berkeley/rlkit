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
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--num_expl_envs", type=int, default=4)
    args = parser.parse_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=10,
            num_trains_per_train_loop=10,
            num_expl_steps_per_train_loop=50,
            min_num_steps_before_training=10,
            num_pretrain_steps=10,
            num_train_loops_per_epoch=1,
            batch_size=30,
            use_wandb=args.use_wandb,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=50,
            num_eval_steps_per_epoch=30,
            num_trains_per_train_loop=200,
            min_num_steps_before_training=5000,
            num_pretrain_steps=100,
            batch_size=625,
            use_wandb=args.use_wandb,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="dreamer_v2",
        version="normal",
        replay_buffer_size=int(1e6),
        algorithm_kwargs=algorithm_kwargs,
        env_class="hinge_cabinet",
        env_kwargs=dict(
            dense=False,
            delta=0.3,
            image_obs=True,
            fixed_schema=True,
            multitask=False,
            action_scale=1.4,
        ),
        vf_kwargs=dict(
            num_layers=3,
        ),
        actor_kwargs=dict(
            discrete_continuous_dist=False,
        ),
        model_kwargs=dict(
            model_hidden_size=400,
            stochastic_state_size=60,
            deterministic_state_size=400,
            gru_layer_norm=False,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            reward_scale=1.0,
            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=6e-4,
            use_amp=True,
            opt_level="O1",
            gradient_clip=100.0,
            lam=0.95,
            free_nats=3.0,
            optimizer_class="apex_adam",
            kl_loss_scale=1.0,
            image_loss_scale=1.0,
            reward_loss_scale=1.0,
            pred_discount_loss_scale=10.0,
            transition_loss_scale=0.0,
            entropy_loss_scale=0.0,
            use_pred_discount=True,
            target_update_period=1,
        ),
        num_expl_envs=args.num_expl_envs,
        num_eval_envs=1,
        expl_amount=0.3,
        path_length_specific_discount=True,
    )

    search_space = {
        "env_class": [
            "microwave",
            "kettle",
            "top_left_burner",
            "slide_cabinet",
            "hinge_cabinet",
            "light_switch",
        ],
        "env_kwargs.delta": [
            0.3,
        ],
        "env_kwargs.fixed_schema": [True, False],
        "env_kwargs.use_combined_action_space": [True, False],
        "env_kwargs.discrete_continuous_dist": [True, False],
        # "trainer_kwargs.image_loss_scale": [
        #     1.0,
        #     1.0 / (64 * 64 * 3),
        # ],
        # "trainer_kwargs.pred_discount_loss_scale": [1.0, 10.0],
        # "trainer_kwargs.transition_loss_scale": [0.08, 0.8],
        # "trainer_kwargs.entropy_loss_scale": [0.02, 0.2],
        # "trainer_kwargs.kl_loss_scale": [0.0, 1.0],
        # "trainer_kwargs.reinforce_loss_scale": [0.9, 1.0, 0.0],
        # "trainer_kwargs.dynamics_backprop_loss_scale": [0.1, 1.0],
        # "trainer_kwargs.actor_entropy_loss_schedule": [
        #     "linear(3e-3,3e-4,2.5e4)",
        #     "linear(3e-3,3e-4,5e4)",
        #     "1e-4",
        # ],
        # "trainer_kwargs.actor_lr": [
        #     4e-5,
        #     8e-5,
        #     1e-4,
        # ],
        # "model_kwargs.discrete_latents": [False, True],  # todo: sweep this
        # "trainer_kwargs.target_update_period": [100],
        # "trainer_kwargs.vf_lr": [1e-4],
        # "trainer_kwargs.adam_eps": [1e-5],
        # "trainer_kwargs.weight_decay": [1e-6],
        # "vf_kwargs.num_layers": [4],
        # "model_kwargs.rssm_hidden_size": [600],
        # "model_kwargs.gru_layer_norm": [True],
        # "model_kwargs.reward_num_layers": [4],
        # "model_kwargs.pred_discount_num_layers": [4],
        # "model_kwargs.discrete_latent_size": [32],
        # "trainer_kwargs.world_model_lr": [
        #     2e-4,
        # ],
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
                snapshot_mode="last",  # saving doesn't seem to work with wandb atm
                python_cmd="~/miniconda3/envs/hrl-exp-env/bin/python",
                seed=seed,
                exp_id=exp_id,
            )
