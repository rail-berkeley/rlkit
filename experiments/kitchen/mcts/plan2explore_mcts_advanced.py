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
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=50,
            num_eval_steps_per_epoch=30,
            min_num_steps_before_training=5000,
            num_pretrain_steps=100,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="Plan2ExploreMCTS",
        version="normal",
        replay_buffer_size=int(1e6),
        algorithm_kwargs=algorithm_kwargs,
        env_kwargs=dict(
            dense=False,
            image_obs=True,
            fixed_schema=False,
            multitask=False,
            action_scale=1.4,
            use_combined_action_space=True,
        ),
        actor_kwargs=dict(
            mean_scale=5.0,
            init_std=5.0,
            use_tanh_normal=True,
            use_per_primitive_actor=False,
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
            use_per_primitive_feature_extractor=False,
        ),
        one_step_ensemble_kwargs=dict(
            num_models=5,
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
            policy_gradient_loss_scale=1.0,
            actor_entropy_loss_schedule="linear(3e-3,3e-4,5e4)",
        ),
        num_expl_envs=args.num_expl_envs,
        num_eval_envs=1,
        path_length_specific_discount=True,
        use_mcts_policy=True,
        expl_policy_kwargs=dict(),
        eval_policy_kwargs=dict(
            randomly_sample_discrete_actions=False,
        ),
        reward_type="intrinsic+extrinsic",
        randomly_sample_discrete_actions=False,
        mcts_algorithm=True,
        trainer_class="plan2explore_advanced_mcts",
        mcts_kwargs=dict(
            mcts_iterations=100,
            dirichlet_alpha=0.03,
            progressive_widening_constant=0.0,
            use_dirichlet_exploration_noise=False,
            use_puct=False,
            normalize_q=False,
            use_reward_discount_value=False,
            use_muzero_uct=False,
            use_max_visit_count=False,
        ),
    )
    search_space = {
        "env_class": [
            "slide_cabinet",
            "microwave",
            "top_left_burner",
            # "kettle",
            # "hinge_cabinet",
            # "light_switch",
        ],
        # "path_length_specific_discount": [True, False],
        # "reward_type": ["intrinsic", "intrinsic+extrinsic", "extrinsic"],
        "mcts_kwargs.dirichlet_alpha": [
            10
        ],
        "mcts_kwargs.progressive_widening_constant": [2.5, 5, 7.5, 10],
        # "mcts_kwargs.normalize_q":[True, False],
        "mcts_kwargs.use_reward_discount_value": [True],
        "mcts_kwargs.use_muzero_uct": [False],
        "mcts_kwargs.use_puct": [True],
        # "mcts_kwargs.use_max_visit_count":[True, False],
        "mcts_kwargs.use_dirichlet_exploration_noise":[True],
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
