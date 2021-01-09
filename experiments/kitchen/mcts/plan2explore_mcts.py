import argparse
import random

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant,
)
from rlkit.torch.model_based.plan2explore.experiments.kitchen_plan2explore_mcts import (
    experiment,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--num_expl_envs", type=int, default=1)
    args = parser.parse_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=10,
            num_trains_per_train_loop=10,
            num_expl_steps_per_train_loop=50,
            min_num_steps_before_training=10,
            num_pretrain_steps=1,
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
        algorithm="p2exp_mcts",
        version="normal",
        replay_buffer_size=int(1e6),
        algorithm_kwargs=algorithm_kwargs,
        # env_class="hinge_cabinet",
        env_kwargs=dict(fixed_schema=False, delta=0.0, dense=False, image_obs=True),
        vf_kwargs=dict(
            num_layers=3,
        ),
        actor_kwargs=dict(
            mean_scale=5.0,
            init_std=5.0,
            use_tanh_normal=True,
            use_per_primitive_actor=False,
            discrete_continuous_dist=True,
        ),
        model_kwargs=dict(
            model_hidden_size=400,
            stochastic_state_size=60,
            deterministic_state_size=400,
            gru_layer_norm=False,
            embedding_size=1024,
            use_per_primitive_feature_extractor=False,
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
            optimizer_class="apex_adam",
            kl_loss_scale=1.0,
            use_pred_discount=True,
            policy_gradient_loss_scale=1.0,
            actor_entropy_loss_schedule="linear(3e-3,3e-4,5e4)",
            mcts_iterations=1000,
        ),
        num_expl_envs=args.num_expl_envs,
        num_eval_envs=1,
        path_length_specific_discount=True,
        use_mcts_policy=True,
        expl_policy_kwargs=dict(
            exploration_weight=0.1,
            parallelize=False,
            exploration=True,
            open_loop_plan=True,
        ),
        eval_policy_kwargs=dict(
            exploration_weight=0.1,
            parallelize=False,
            exploration=False,
            open_loop_plan=False,
        ),
        one_step_ensemble_kwargs=dict(
            num_models=5,
            hidden_size=400,
            num_layers=4,
            output_embeddings=False,
        ),
        mcts_iterations=10000,
        randomly_sample_discrete_actions=True,
    )

    search_space = {
        "env_class": [
            "microwave",
            "top_left_burner",
            "slide_cabinet",
            # "kettle",
            # "hinge_cabinet",
            # "light_switch",
        ],
        "expl_policy_kwargs.open_loop_plan": [True],
        "eval_policy_kwargs.open_loop_plan": [False],
        "env_kwargs.delta": [0.1, 0.3, 0.5],
        "randomly_sample_discrete_actions": [True],
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
