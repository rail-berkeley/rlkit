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
            num_eval_steps_per_epoch=10,
            num_expl_steps_per_train_loop=50,
            min_num_steps_before_training=10,
            num_pretrain_steps=10,
            num_train_loops_per_epoch=1,
            num_trains_per_train_loop=10,
            batch_size=30,
            max_path_length=5,
            use_pretrain_policy_for_initial_data=False,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=25,
            num_eval_steps_per_epoch=30,
            min_num_steps_before_training=2500,
            num_pretrain_steps=100,
            max_path_length=5,
            batch_size=417,  # 417*6 = 2502
            num_expl_steps_per_train_loop=30,  # 5*(5+1) one trajectory per vec env
            num_train_loops_per_epoch=40,  # 1000//(5*5)
            num_trains_per_train_loop=5,  # 200//40
            use_pretrain_policy_for_initial_data=False,
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="DreamerV2",
        version="normal",
        replay_buffer_size=int(5e5),
        algorithm_kwargs=algorithm_kwargs,
        env_class="hinge_cabinet",
        env_kwargs=dict(
            dense=False,
            image_obs=True,
            fixed_schema=False,
            action_scale=1.4,
            use_combined_action_space=True,
            proprioception=False,
            wrist_cam_concat_with_fixed_view=False,
            use_wrist_cam=False,
            normalize_proprioception_obs=True,
            use_workspace_limits=True,
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
            discount=0.99,
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
        ),
        num_expl_envs=5,
        num_eval_envs=1,
        expl_amount=0.3,
        load_from_path=True,
        # models_path="/home/mdalal/research/rlkit/data/02-10-p2exp-sc-expl-with-eval-actor-v1/02-10-p2exp_sc_expl_with_eval_actor_v1_2021_02_10_13_55_48_0001--s-11474/",
        retrain_actor_and_vf=False,
        pkl_file_name="/itr_400.pkl",
    )

    search_space = {
        "env_class": [
            "microwave",
            "kettle",
            "slide_cabinet",
            "top_left_burner",
            "hinge_cabinet",
            "light_switch",
        ],
        "trainer_kwargs.discount": [0.8],
        "retrain_actor_and_vf": [False],
        # "algorithm_kwargs.use_pretrain_policy_for_initial_data": [False],
        "num_actor_vf_pretrain_iters": [1000, 10000],
        "algorithm_kwargs.use_pretrain_policy_for_initial_data": [True],
        "algorithm_kwargs.num_pretrain_steps": [0],
        "algorithm_kwargs.num_trains_per_train_loop": [50],
        "trainer_kwargs.world_model_lr": [3e-4],
        "pkl_file_name": [
            "/itr_100.pkl",
            # "/itr_200.pkl",
            # "/itr_300.pkl",
            # "/itr_400.pkl",
        ]
        # "models_path": [
        #     os.path.join(
        #         "/home/mdalal/research/rlkit/data/02-10-p2exp-sc-expl-with-eval-actor-v1/",
        #         path,
        #     )
        #     for path in os.listdir(
        #         "/home/mdalal/research/rlkit/data/02-10-p2exp-sc-expl-with-eval-actor-v1/"
        #     )
        # ]
        # "env_kwargs.use_workspace_limits": [True, False],
        # "trainer_kwargs.actor_entropy_loss_schedule": ["linear(3e-3,3e-4,5e4)", "1e-4"],
    }
    models_path_dict = {
        "microwave": [
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-14270",
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-2491",
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-78811",
        ],
        "kettle": [
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-12648",
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-71818",
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-92834",
        ],
        "slide_cabinet": [
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-59167",
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-75453",
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-8782",
        ],
        "top_left_burner": [
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-34797",
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-57354",
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-68791",
        ],
        "hinge_cabinet": [
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-40385",
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_31_0000--s-61206",
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-14169",
        ],
        "light_switch": [
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-21596",
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-2217",
            "/home/mdalal/research/rlkit/data/02-10-kitchen-p2exp-intrinsic-run-long-v1/02-10-kitchen_p2exp_intrinsic_run_long_v1_2021_02_10_20_55_32_0000--s-38660",
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    num_exps_launched = 0
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant = preprocess_variant(variant, args.debug)
        for s in range(len(models_path_dict[variant["env_class"]])):
            variant["models_path"] = models_path_dict[variant["env_class"]][s]
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
