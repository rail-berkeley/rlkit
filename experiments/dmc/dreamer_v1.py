import random

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.arguments import get_args
from rlkit.torch.model_based.dreamer.experiments.dmc_dreamer_experiment import (
    experiment,
)

if __name__ == "__main__":
    args = get_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=1000,
            min_num_steps_before_training=1000,
            num_pretrain_steps=1,
            max_path_length=500,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1,
            num_train_loops_per_epoch=1,
            batch_size=50,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=200,
            num_eval_steps_per_epoch=2500,
            min_num_steps_before_training=2500,
            num_pretrain_steps=100,
            max_path_length=500,
            num_expl_steps_per_train_loop=500,
            num_trains_per_train_loop=100,
            num_train_loops_per_epoch=20,
            batch_size=50,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="Dreamer",
        version="normal",
        replay_buffer_size=int(1e3),
        algorithm_kwargs=algorithm_kwargs,
        actor_kwargs=dict(
            use_tanh_normal=True,
            mean_scale=5.0,
            init_std=5.0,
            num_layers=4,
            dist="tanh_normal_dreamer_v1",
        ),
        vf_kwargs=dict(
            num_layers=3,
        ),
        model_kwargs=dict(
            model_hidden_size=400,
            stochastic_state_size=30,
            deterministic_state_size=200,
            embedding_size=1024,
            use_per_primitive_feature_extractor=False,
            rssm_hidden_size=200,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=6e-4,
            lam=0.95,
            free_nats=3.0,
            kl_loss_scale=1.0,
            imagination_horizon=15,
            use_pred_discount=False,
        ),
        num_eval_envs=1,
        expl_amount=0.0,
    )

    search_space = {
        "env_id": [
            # "walker_walk",
            # "pendulum_swingup",
            # "cartpole_swingup",
            # "hopper_stand",
            # "walker_run",
            "quadruped_walk",
            # "acrobot_swingup",
            # "hopper_hop",
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    num_exps_launched = 0
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
            )
            num_exps_launched += 1
    print("Num exps launched: ", num_exps_launched)
