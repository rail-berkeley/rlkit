import json
import os
import random
import subprocess
import time

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.arguments import get_args
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant,
)
from rlkit.torch.model_based.dreamer.experiments.raps_experiment import experiment

if __name__ == "__main__":
    args = get_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=1000,
            min_num_steps_before_training=1000,
            num_pretrain_steps=1,
            max_path_length=280,
            num_expl_steps_per_train_loop=281,
            num_trains_per_train_loop=1,
            num_train_loops_per_epoch=1,
            batch_size=50,
            use_pretrain_policy_for_initial_data=False,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=100,
            num_eval_steps_per_epoch=280 * 5,
            min_num_steps_before_training=2500,
            num_pretrain_steps=100,
            max_path_length=280,
            batch_size=50,  # 417*6 = 2502
            num_expl_steps_per_train_loop=281 * 5,  # 5*(5+1) one trajectory per vec env
            num_trains_per_train_loop=572,  # 200//40
            num_train_loops_per_epoch=7,  # 1000//(5*5)
            use_pretrain_policy_for_initial_data=False,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="DreamerV2",
        version="normal",
        replay_buffer_size=int(9e3),
        algorithm_kwargs=algorithm_kwargs,
        use_raw_actions=True,
        env_kwargs=dict(
            dense=False,
            image_obs=True,
            action_scale=1,
            proprioception=False,
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
            detach_rewards=False,
        ),
        num_expl_envs=5,
        num_eval_envs=1,
        expl_amount=0.3,
        load_from_path=True,
        retrain_actor_and_vf=False,
        pkl_file_name="/itr_30.pkl",
    )

    search_space = {
        "env_name": [
            # "microwave",
            # "kettle",
            # "slide_cabinet",
            # "top_left_burner",
            "hinge_cabinet",
            # "light_switch",
        ],
        "retrain_actor_and_vf": [False],
        "num_actor_vf_pretrain_iters": [1000],
        "algorithm_kwargs.use_pretrain_policy_for_initial_data": [True],
        "algorithm_kwargs.num_pretrain_steps": [0],
        "trainer_kwargs.world_model_lr": [3e-4],
        "pkl_file_name": [
            "/params.pkl",
        ],
    }
    models_path_dict = {
        # "microwave": [
        #     "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-14270",
        #     "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-2491",
        #     "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-78811",
        # ],
        # "kettle": [
        #     "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-12648",
        #     "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-71818",
        #     "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-92834",
        # ],
        "hinge_cabinet": [
            "/home/mdalal/research/rlkit/data/05-16-kitchen-p2exp-raw-actions-hinge-v1/05-16-kitchen_p2exp_raw_actions_hinge_v1_2021_05_16_12_51_24_0000--s-55945/",
            "/home/mdalal/research/rlkit/data/05-16-kitchen-p2exp-raw-actions-hinge-v1/05-16-kitchen_p2exp_raw_actions_hinge_v1_2021_05_16_12_51_24_0000--s-73949/",
            "/home/mdalal/research/rlkit/data/05-16-kitchen-p2exp-raw-actions-hinge-v1/05-16-kitchen_p2exp_raw_actions_hinge_v1_2021_05_16_12_51_24_0000--s-54978/",
        ],
        # "top_left_burner": [
        #     "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-34797",
        #     "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-57354",
        #     "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-68791",
        # ],
        # "hinge_cabinet": [
        #     "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-40385",
        #     "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-61206",
        #     "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-14169",
        # ],
        # "light_switch": [
        #     "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-21596",
        #     "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-2217",
        #     "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-38660",
        # ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    num_exps_launched = 0
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant = preprocess_variant(variant, args.debug)
        for s in range(len(models_path_dict[variant["env_name"]])):
            variant["models_path"] = models_path_dict[variant["env_name"]][s]
            models_path = variant["models_path"]
            seed = int(json.load(open(models_path + "/variant.json", "r"))["seed"])
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
            time.sleep(1)
            num_exps_launched += 1
    print("Num exps launched: ", num_exps_launched)
