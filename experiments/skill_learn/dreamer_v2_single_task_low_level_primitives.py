import argparse
import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant,
)
from rlkit.torch.model_based.dreamer.experiments.low_level_primitives_exp import (
    experiment,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=5,
            num_expl_steps_per_train_loop=5,
            min_num_steps_before_training=0,
            num_pretrain_steps=10,
            num_train_loops_per_epoch=1,
            num_trains_per_train_loop=10,
            batch_size=25,
            max_path_length=5,
        )
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=30 * 2,
            min_num_steps_before_training=0,
            num_pretrain_steps=1000,
            max_path_length=5,
            batch_size=25,
            num_expl_steps_per_train_loop=60,  # 5*(5+1) one trajectory per vec env
            num_train_loops_per_epoch=20,  # 1000//(5*5)
            num_trains_per_train_loop=20,  # 400//40
        )
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="DreamerV2",
        version="normal",
        replay_buffer_size=int(12e3),
        algorithm_kwargs=algorithm_kwargs,
        use_raw_actions=False,
        env_suite="metaworld",
        pass_render_kwargs=True,
        env_kwargs=dict(
            control_mode="primitives",
            action_scale=1,
            max_path_length=5,
            reward_type="sparse",
            camera_settings={
                "distance": 0.38227044687537043,
                "lookat": [0.21052547, 0.32329237, 0.587819],
                "azimuth": 141.328125,
                "elevation": -53.203125160653144,
            },
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=True,
                max_path_length=5,
                unflatten_images=False,
            ),
            image_kwargs=dict(imwidth=64, imheight=64),
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
            use_prior_instead_of_posterior=True,
        ),
        trainer_kwargs=dict(
            adam_eps=1e-8,
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
            batch_length=100,
            weight_decay=0.0,
        ),
        num_expl_envs=5 * 2,
        num_eval_envs=1,
        expl_amount=0.3,
        save_video=True,
        low_level_action_dim=9,
        mlp_hidden_sizes=[512, 512],
    )

    search_space = {
        "env_name": [
            # "assembly-v2",
            # "disassemble-v2",
            # "peg-unplug-side-v2",
            # "sweep-into-v2",
            # "soccer-v2",
            "drawer-close-v2",
        ],
        "algorithm_kwargs.num_train_loops_per_epoch": [10],
        "algorithm_kwargs.num_expl_steps_per_train_loop": [0],
        "algorithm_kwargs.num_pretrain_steps": [1500],
        "algorithm_kwargs.num_trains_per_train_loop": [400],
        "num_low_level_actions_per_primitive": [50],
        "num_trajs": [100],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant["replay_buffer_path"] = (
            "wm_H_5_T_{}_E_50_P_{}_raps_ll_hl_even_rt_{}".format(
                variant["num_trajs"],
                variant["num_low_level_actions_per_primitive"],
                variant["env_name"],
            )
            + ".hdf5"
        )
        variant["replay_buffer_path"] = (
            "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/"
            + variant["replay_buffer_path"]
        )
        if args.debug:
            variant["algorithm_kwargs"]["num_pretrain_steps"] = 1
        if variant["env_name"] == "soccer-v2":
            variant[
                "world_model_path"
            ] = "/home/mdalal/research/skill_learn/rlkit/data/11-24-train-wm-with-prims-sweep-envs-1/11-24-train_wm_with_prims_sweep_envs_1_2021_11_24_19_23_14_0000--s-96844/models/world_model.pt"
        elif variant["env_name"] == "drawer-close-v2":
            variant[
                "world_model_path"
            ] = "/home/mdalal/research/skill_learn/rlkit/data/12-07-train-wm-with-prims-sweep-drawer-subsample-more-data-sweep-1/12-07-train_wm_with_prims_sweep_drawer_subsample_more_data_sweep_1_2021_12_07_12_52_01_0000--s-75299/models/world_model.pt"
        elif variant["env_name"] == "sweep-into-v2":
            variant[
                "world_model_path"
            ] = "/home/mdalal/research/skill_learn/rlkit/data/11-27-train-wm-with-prims-sweep-envs-all-1/11-27-train_wm_with_prims_sweep_envs_all_1_2021_11_27_23_16_29_0000--s-93127/models/world_model.pt"
        elif variant["env_name"] == "peg-unplug-side-v2":
            variant[
                "world_model_path"
            ] = "/home/mdalal/research/skill_learn/rlkit/data/11-27-train-wm-with-prims-sweep-envs-all-1/11-27-train_wm_with_prims_sweep_envs_all_1_2021_11_27_23_16_29_0000--s-44001/models/world_model.pt"
        elif variant["env_name"] == "disassemble-v2":
            variant[
                "world_model_path"
            ] = "/home/mdalal/research/skill_learn/rlkit/data/11-27-train-wm-with-prims-sweep-envs-all-1/11-27-train_wm_with_prims_sweep_envs_all_1_2021_11_27_23_16_29_0000--s-19641/models/world_model.pt"
        elif variant["env_name"] == "assembly-v2":
            # TODO: fix with real path
            variant[
                "world_model_path"
            ] = "/home/mdalal/research/skill_learn/rlkit/data/11-27-train-wm-with-prims-sweep-envs-all-1/11-27-train_wm_with_prims_sweep_envs_all_1_2021_11_27_23_16_29_0000--s-19641/models/world_model.pt"
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
