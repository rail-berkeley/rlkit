import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.arguments import get_args
from rlkit.torch.model_based.plan2explore.experiments.dmc_plan2explore_experiment import (
    experiment,
)

if __name__ == "__main__":
    args = get_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=1000,
            min_num_steps_before_training=1000,
            num_pretrain_steps=1,
            max_path_length=500,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1,
            num_train_loops_per_epoch=1,
            batch_size=2,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=50,
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
        algorithm="Plan2Explore",
        version="normal",
        replay_buffer_size=int(1e4),
        algorithm_kwargs=algorithm_kwargs,
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
            stochastic_state_size=32,
            deterministic_state_size=200,
            embedding_size=32 * 48,
            rssm_hidden_size=200,
            reward_num_layers=2,
            pred_discount_num_layers=3,
            gru_layer_norm=True,
            std_act="sigmoid2",
            depth=48,
        ),
        one_step_ensemble_kwargs=dict(
            num_models=10,
            hidden_size=400,
            num_layers=4,
            inputs="feat",
            targets="stoch",
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
            log_disagreement=True,
            detach_rewards=True,
            ensemble_training_states="post_to_next_post",
        ),
        num_eval_envs=1,
        expl_amount=0.0,
    )

    search_space = {
        "env_id": [
            "walker_walk",
            # "pendulum_swingup",
            # "cartpole_swingup_sparse",
            # "cartpole_swingup",
            # "cartpole_balance",
            # "cartpole_balance_sparse",
            # "hopper_stand",
            # "walker_run",
            # "quadruped_walk",
            # "acrobot_swingup",
            # "hopper_hop",
            # "cheetah_run",
        ],
        # "one_step_ensemble_kwargs.inputs": ["feats", "deter", "stoch"],
        # "one_step_ensemble_kwargs.targets": ["feats", "deter", "stoch", "embed"],
        # "trainer_kwargs.ensemble_training_states": [
        #     "post_to_next_post",
        #     "post_to_next_prior",
        #     "prior_to_next_post",
        #     "prior_to_next_prior",
        # ],
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
