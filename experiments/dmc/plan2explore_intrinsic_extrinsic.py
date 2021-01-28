import argparse
import random

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.plan2explore.experiments.dmc_plan2explore import experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=1000,
            min_num_steps_before_training=1000,
            num_pretrain_steps=1,
            max_path_length=1000 // 2,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1,
            num_train_loops_per_epoch=1,
            batch_size=50,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=5000,
            min_num_steps_before_training=5000,
            num_pretrain_steps=100,
            max_path_length=1000 // 2,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            num_train_loops_per_epoch=10,
            batch_size=50,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="Plan2Explore",
        version="normal",
        replay_buffer_size=int(1e3),
        algorithm_kwargs=algorithm_kwargs,
        actor_kwargs=dict(
            discrete_continuous_dist=True,
            use_per_primitive_actor=False,
            use_tanh_normal=True,
            mean_scale=5.0,
            init_std=5.0,
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
            policy_gradient_loss_scale=0.0,
            actor_entropy_loss_schedule="0.0",
            train_decoder_on_second_output_only=False,
            use_next_feat_for_computing_reward=False,
            one_step_ensemble_pred_prior_from_prior=True,
            imagination_horizon=15,
            train_exploration_actor_with_intrinsic_and_extrinsic_reward=True,
            train_actor_with_intrinsic_and_extrinsic_reward=True,
            exploration_reward_scale=0.0,
            detach_rewards=False,
            use_pred_discount=False,
        ),
        num_eval_envs=1,
        expl_amount=0.3,
        reward_type="intrinsic",
    )

    search_space = {
        "env_id": [
            "acrobot_swingup",
            "pendulum_swingup",
            "quadruped_walk",
            "walker_run",
            "walker_walk",
            "hopper_hop",
            "hopper_stand",
            "cartpole_swingup",
        ],
        "expl_amount": [0.3],
        "reward_type": ["extrinsic", "intrinsic", "intrinsic+extrinsic"],
        "trainer_kwargs.train_decoder_on_second_output_only": [True, False],
        "model_kwargs.embedding_size": [
            1024,
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
                python_cmd="~/miniconda3/envs/hrl-exp-env/bin/python",
                seed=seed,
                exp_id=exp_id,
            )
            num_exps_launched += 1
    print("Num exps launched: ", num_exps_launched)
