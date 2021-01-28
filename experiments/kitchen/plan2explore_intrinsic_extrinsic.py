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
            num_epochs=50,
            num_eval_steps_per_epoch=30,
            num_trains_per_train_loop=200,
            min_num_steps_before_training=5000,
            num_pretrain_steps=100,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="Plan2Explore",
        version="normal",
        replay_buffer_size=int(1e5),
        algorithm_kwargs=algorithm_kwargs,
        env_kwargs=dict(
            dense=False,
            image_obs=True,
            fixed_schema=False,
            multitask=False,
            action_scale=1.4,
            use_combined_action_space=True,
            proprioception=False,
            wrist_cam_concat_with_fixed_view=False,
            use_wrist_cam=False,
            normalize_proprioception_obs=True,
        ),
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
            stochastic_state_size=60,
            deterministic_state_size=400,
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
            pred_discount_loss_scale=10.0,
            policy_gradient_loss_scale=1.0,
            actor_entropy_loss_schedule="linear(3e-3,3e-4,5e4)",
            state_loss_scale=1.0,
            train_decoder_on_second_output_only=False,
            use_next_feat_for_computing_reward=False,
            one_step_ensemble_pred_prior_from_prior=True,
        ),
        num_expl_envs=args.num_expl_envs,
        num_eval_envs=1,
        expl_amount=0.3,
        path_length_specific_discount=True,
        eval_with_exploration_actor=False,
        reward_type="intrinsic",
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
        "expl_amount": [0.3],
        # "env_kwargs.use_combined_action_space": [False],
        # "env_kwargs.delta": [0.3],
        # "trainer_kwargs.use_next_feat_for_computing_reward": [True, False],
        # "trainer_kwargs.one_step_ensemble_pred_prior_from_prior": [True, False],
        "reward_type": ["extrinsic", "intrinsic", "intrinsic+extrinsic"],
        "env_kwargs.proprioception": [True, False],
        "env_kwargs.wrist_cam_concat_with_fixed_view": [True, False],
        "env_kwargs.use_wrist_cam": [True, False],
        "trainer_kwargs.train_decoder_on_second_output_only": [True, False],
        # "env_kwargs.wrist_cam_concat_with_fixed_view": [True],
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
        if (not variant["env_kwargs"]["wrist_cam_concat_with_fixed_view"]) and variant[
            "trainer_kwargs"
        ]["train_decoder_on_second_output_only"]:
            continue
        if (
            variant["env_kwargs"]["use_wrist_cam"]
            and variant["env_kwargs"]["wrist_cam_concat_with_fixed_view"]
        ):
            continue
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
            num_exps_launched += 1
    print("Num exps launched: ", num_exps_launched)
