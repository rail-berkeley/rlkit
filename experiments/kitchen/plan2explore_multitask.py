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
            num_epochs=2,
            num_eval_steps_per_epoch=30,
            num_trains_per_train_loop=10,
            num_expl_steps_per_train_loop=150,  # 200 samples since num_envs = 50 and max_path_length + 1 = 4
            min_num_steps_before_training=100,
            num_pretrain_steps=100,
            num_train_loops_per_epoch=1,
            batch_size=50,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=100,
            num_eval_steps_per_epoch=30,
            num_trains_per_train_loop=200,
            num_expl_steps_per_train_loop=150,  # 200 samples since num_envs = 50 and max_path_length + 1 = 4
            min_num_steps_before_training=5000,
            num_pretrain_steps=100,
            num_train_loops_per_epoch=5,
            batch_size=625,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="Dreamer",
        version="normal",
        replay_buffer_size=int(1e5),
        algorithm_kwargs=algorithm_kwargs,
        env_class="multitask_all",
        env_kwargs=dict(
            dense=False,
            delta=0.0,
            image_obs=True,
            fixed_schema=True,
        ),
        world_model_class="multitask",
        actor_kwargs=dict(
            discrete_continuous_dist=False,
        ),
        model_kwargs=dict(
            model_hidden_size=400,
            stochastic_state_size=60,
            deterministic_state_size=400,
            embedding_size=1031,
        ),
        one_step_ensemble_kwargs=dict(
            hidden_size=32 * 32,
            num_layers=2,
            num_models=5,
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
            kl_loss_scale=1.0,
            optimizer_class="apex_adam",
            pred_discount_loss_scale=10.0,
            use_pred_discount=True,
        ),
        num_expl_envs=args.num_expl_envs,
        num_eval_envs=1,
    )

    search_space = {
        "env_kwargs.delta": [0, 0.1, 0.3, 0.5],
        "env_kwargs.use_combined_action_space": [True, False],
        "expl_amount": [0.3, 0.6, 0.9],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if (
            variant["env_kwargs"]["use_combined_action_space"]
            and variant["env_kwargs"]["delta"] != 0.0
        ):
            continue
        variant = preprocess_variant(variant, args.debug)
        for _ in range(args.num_seeds):
            run_experiment(
                experiment,
                exp_prefix=args.exp_prefix,
                mode=args.mode,
                variant=variant,
                use_gpu=True,
                snapshot_mode="last",
                python_cmd="~/miniconda3/envs/hrl-exp-env/bin/python",
                seed=random.randint(0, 100000),
                exp_id=exp_id,
            )