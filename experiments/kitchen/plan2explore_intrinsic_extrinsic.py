import argparse
import random

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant,
)
from rlkit.torch.model_based.plan2explore.experiments.kitchen_plan2explore import (
    experiment,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
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
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=100,
            num_eval_steps_per_epoch=30,
            num_trains_per_train_loop=200,
            min_num_steps_before_training=5000,
            num_pretrain_steps=100,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="Plan2ExploreInstrinsicExtrinsic",
        version="normal",
        replay_buffer_size=int(1e6),
        algorithm_kwargs=algorithm_kwargs,
        env_class="microwave",
        env_kwargs=dict(
            dense=False,
            delta=0.3,
            image_obs=True,
            fixed_schema=False,
            multitask=False,
            action_scale=1.4,
            use_combined_action_space=False,
        ),
        actor_kwargs=dict(
            discrete_continuous_dist=True,
        ),
        vf_kwargs=dict(
            num_layers=3,
        ),
        model_kwargs=dict(
            model_hidden_size=400,
            stochastic_state_size=60,
            deterministic_state_size=400,
            embedding_size=1024,
            use_depth_wise_separable_conv=False,
        ),
        one_step_ensemble_kwargs=dict(
            num_models=10,
            hidden_size=400,
            num_layers=4,
            output_embeddings=False,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            reward_scale=1.0,
            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=6e-4,
            use_amp=True,
            opt_level="O1",
            lam=0.95,
            free_nats=3.0,
            kl_loss_scale=1.0,
            optimizer_class="apex_adam",
            pred_discount_loss_scale=10.0,
            use_pred_discount=True,
            train_with_intrinsic_and_extrinsic_reward=True,
            policy_gradient_loss_scale=1.0,
            actor_entropy_loss_schedule="linear(3e-3,3e-4,5e4)",
            detach_rewards=False,
        ),
        num_expl_envs=args.num_expl_envs,
        num_eval_envs=1,
        expl_amount=0.3,
        path_length_specific_discount=True,
    )

    search_space = {
        "env_class": [
            # "microwave",
            # "kettle",
            "top_left_burner",
            # "slide_cabinet",
            "hinge_cabinet",
            "light_switch",
        ],
        # "env_kwargs.use_combined_action_space": [True],
        # "env_kwargs.use_max_bound_action_space": [False],
        "trainer_kwargs.exploration_reward_scale": [0.1, .5, 1, 10],
        "expl_amount": [0.3, 0.5],
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
                snapshot_mode="last",
                python_cmd="~/miniconda3/envs/hrl-exp-env/bin/python",
                seed=seed,
                exp_id=exp_id,
            )
