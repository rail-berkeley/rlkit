import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.arguments import get_args
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant,
)
from rlkit.torch.model_based.plan2explore.experiments.raps_plan2explore_experiment import (
    experiment,
)

if __name__ == "__main__":
    args = get_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=5,
            num_expl_steps_per_train_loop=500,
            min_num_steps_before_training=100,
            num_pretrain_steps=10,
            num_train_loops_per_epoch=1,
            num_trains_per_train_loop=10,
            batch_size=50,
            max_path_length=280,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=50,
            num_eval_steps_per_epoch=280 * 5,
            min_num_steps_before_training=2500,
            num_pretrain_steps=100,
            max_path_length=280,
            num_expl_steps_per_train_loop=281 * 5,
            num_trains_per_train_loop=285,
            num_train_loops_per_epoch=7,
            batch_size=50,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="Plan2Explore",
        version="normal",
        replay_buffer_size=int(9e3),
        algorithm_kwargs=algorithm_kwargs,
        num_expl_envs=5,
        num_eval_envs=1,
        expl_amount=0.3,
        save_video=False,
        use_raw_actions=True,
        pass_render_kwargs=False,
        env_suite="kitchen",
        env_kwargs=dict(
            dense=False,
            image_obs=True,
            action_scale=1,
            use_workspace_limits=True,
            control_mode="joint_velocity",
            frame_skip=40,
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=True,
                max_path_length=280,
                unflatten_images=False,
            ),
            image_kwargs=dict(),
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
            log_disagreement=False,
            ensemble_training_states="post_to_next_post",
            detach_rewards=True,
        ),
        reward_type="intrinsic",
        eval_with_exploration_actor=False,
        expl_with_exploration_actor=True,
        actor_kwargs=dict(
            init_std=0.0,
            num_layers=4,
            min_std=0.1,
            dist="trunc_normal",
            discrete_continuous_dist=False,
        ),
    )

    search_space = {
        "env_name": [
            # "microwave",
            # "top_left_burner",
            "hinge_cabinet",
            # "light_switch",
            # "slide_cabinet",
            # "kettle",
        ],
        "one_step_ensemble_kwargs.inputs": ["deter"],
        "one_step_ensemble_kwargs.targets": ["stoch"],
        "trainer_kwargs.ensemble_training_states": [
            "prior_to_next_post",
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    num_exps_launched = 0
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
                snapshot_mode="gap_and_last",
                snapshot_gap=10,
                python_cmd=subprocess.check_output("which python", shell=True).decode(
                    "utf-8"
                )[:-1],
                seed=seed,
                exp_id=exp_id,
            )
            num_exps_launched += 1
    print("Num exps launched: ", num_exps_launched)
