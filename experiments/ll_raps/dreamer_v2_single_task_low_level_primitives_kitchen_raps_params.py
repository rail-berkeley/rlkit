import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.arguments import get_args
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant,
)
from rlkit.torch.model_based.dreamer.experiments.ll_raps_experiment import experiment

if __name__ == "__main__":
    args = get_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=10,
            num_expl_steps_per_train_loop=50,
            min_num_steps_before_training=10,
            num_pretrain_steps=10,
            num_train_loops_per_epoch=1,
            num_trains_per_train_loop=10,
            batch_size=200,
            max_path_length=5,
        )
    else:
        algorithm_kwargs = dict(
            num_epochs=250,
            num_eval_steps_per_epoch=30,
            min_num_steps_before_training=2500,
            num_pretrain_steps=100,
            max_path_length=5,
            batch_size=200,
            num_expl_steps_per_train_loop=30 * 2,  # 5*(5+1) one trajectory per vec env
            num_train_loops_per_epoch=40 // 2,  # 1000//(5*5)
            num_trains_per_train_loop=10 * 2,  # 400//40
        )
    variant = dict(
        algorithm="LLRAPS",
        version="normal",
        replay_buffer_size=int(1.2e4),
        algorithm_kwargs=algorithm_kwargs,
        use_raw_actions=False,
        env_suite="kitchen",
        pass_render_kwargs=True,
        env_kwargs=dict(
            dense=False,
            image_obs=True,
            action_scale=1.4,
            use_workspace_limits=True,
            control_mode="primitives",
            num_low_level_actions_per_primitive=10,
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=True,
                max_path_length=5,
                unflatten_images=False,
            ),
            image_kwargs=dict(),
            collect_primitives_info=True,
            render_intermediate_obs_to_info=True,
        ),
        actor_kwargs=dict(
            discrete_continuous_dist=True,
            init_std=0.0,
            num_layers=4,
            min_std=0.1,
            dist="tanh_normal_dreamer_v1",
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
        trainer_kwargs=dict(
            adam_eps=1e-5,
            discount=0.8,
            lam=0.95,
            forward_kl=False,
            free_nats=1.0,
            pred_discount_loss_scale=10.0,
            kl_loss_scale=0.0,
            transition_loss_scale=0.8,
            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=3e-4,
            reward_loss_scale=2.0,
            use_pred_discount=True,
            policy_gradient_loss_scale=1.0,
            actor_entropy_loss_schedule="1e-4",
            target_update_period=100,
            detach_rewards=False,
            imagination_horizon=5,
            wm_loss_scale=-1,
        ),
        num_expl_envs=1,
        num_eval_envs=1,
        expl_amount=0.3,
        save_video=True,
        low_level_action_dim=9,
        mlp_hidden_sizes=[512, 512],
        prioritize_fraction=0.0,
    )
    search_space = {
        key: value for key, value in zip(args.search_keys, args.search_values)
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if args.debug:
            variant["algorithm_kwargs"]["num_pretrain_steps"] = 1
            variant["algorithm_kwargs"]["min_num_steps_before_training"] = 10
            variant["algorithm_kwargs"]["num_trains_per_train_loop"] = 1
        variant["replay_buffer_size"] = int(
            3e6 / (variant["num_low_level_actions_per_primitive"] * 5 + 1)
        )
        variant["trainer_kwargs"]["batch_length"] = int(
            variant["num_low_level_actions_per_primitive"] * 5 + 1
        )
        variant["env_kwargs"]["num_low_level_actions_per_primitive"] = variant[
            "num_low_level_actions_per_primitive"
        ]
        variant["trainer_kwargs"]["num_world_model_training_iterations"] = (
            400 // variant["algorithm_kwargs"]["batch_size"]
        )
        if variant["trainer_kwargs"]["wm_loss_scale"] == -1:
            variant["trainer_kwargs"]["wm_loss_scale"] = 1 / (
                variant["trainer_kwargs"]["num_world_model_training_iterations"]
            )
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
                python_cmd=subprocess.check_output("which python", shell=True).decode(
                    "utf-8"
                )[:-1],
                seed=seed,
                exp_id=exp_id,
            )
