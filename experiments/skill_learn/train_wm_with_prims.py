import argparse
import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.wm_prim_exp import experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    if args.debug:
        algorithm_kwargs = dict()
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict()
        exp_prefix = args.exp_prefix
    variant = dict(
        low_level_primitives=True,
        num_low_level_actions_per_primitive=100,
        low_level_action_dim=9,
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
            include_phase_variable=True,
            render_intermediate_obs_to_info=True,
        ),
        env_suite="metaworld",
        env_name="reach-v2",
        world_model_loss_kwargs=dict(
            forward_kl=False,
            free_nats=1.0,
            transition_loss_scale=0.8,
            kl_loss_scale=0.0,
            image_loss_scale=1.0,
            discount=0.8,
            reward_loss_scale=2.0,
            pred_discount_loss_scale=10.0,
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
        optimizer_kwargs=dict(
            weight_decay=0.0,
        ),
        dataloader_kwargs=dict(
            batch_size=25,
            train_test_split=0.8,
        ),
        mlp_hidden_sizes=(512, 512),
        clone_primitives=False,
        clone_primitives_separately=False,
        plotting_period=250,
        num_epochs=10000,
        visualize_wm_from_path=False,
        clone_primitives_and_train_world_model=True,
        mlp_act="relu",
        mlp_res=False,
    )

    search_space = {
        # "datafile": [
        #     "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_100_raps_ll_hl_even_rt_drawer-close-v2.hdf5",
        #     # "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_100_raps_ll_hl_even_rt_soccer-v2.hdf5",
        #     # "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_100_raps_ll_hl_even_rt_sweep-into-v2.hdf5",
        #     # "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_100_raps_ll_hl_even_rt_peg-unplug-side-v2.hdf5",
        #     # "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_100_raps_ll_hl_even_rt_disassemble-v2.hdf5",
        #     # "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_100_raps_ll_hl_even_rt_assembly-v2.hdf5",
        # ],
        "datafile": [
            "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_100_raps_ll_hl_even_rt_drawer-close-v2.hdf5",
            "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_50_raps_ll_hl_even_rt_drawer-close-v2.hdf5",
            "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_25_raps_ll_hl_even_rt_drawer-close-v2.hdf5",
            "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_10_raps_ll_hl_even_rt_drawer-close-v2.hdf5",
        ],
        "batch_len": [100],
        "dataloader_kwargs.batch_size": [25],
        "optimizer_kwargs.lr": [3e-4, 1e-3],
        "optimizer_kwargs.eps": [1e-8],
        "loss_to_use": ["both"],
        "gradient_clip": [100],
        # "mlp_act": ["elu", "relu"],
        # "mlp_res": [True, False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if (
            variant["datafile"]
            == "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_100_raps_ll_hl_even_rt_drawer-close-v2.hdf5"
        ):
            variant["env_name"] = "drawer-close-v2"
            variant["batch_len"] = 100
            variant["dataloader_kwargs"]["batch_size"] = 25
            variant["num_low_level_actions_per_primitive"] = 100
        elif (
            variant["datafile"]
            == "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_50_raps_ll_hl_even_rt_drawer-close-v2.hdf5"
        ):
            variant["env_name"] = "drawer-close-v2"
            variant["batch_len"] = 100
            variant["dataloader_kwargs"]["batch_size"] = 25
            variant["num_low_level_actions_per_primitive"] = 50
        elif (
            variant["datafile"]
            == "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_25_raps_ll_hl_even_rt_drawer-close-v2.hdf5"
        ):
            variant["env_name"] = "drawer-close-v2"
            variant["batch_len"] = 100
            variant["dataloader_kwargs"]["batch_size"] = 25
            variant["num_low_level_actions_per_primitive"] = 25
        elif (
            variant["datafile"]
            == "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_10_raps_ll_hl_even_rt_drawer-close-v2.hdf5"
        ):
            variant["env_name"] = "drawer-close-v2"
            variant["batch_len"] = 50
            variant["dataloader_kwargs"]["batch_size"] = 50
            variant["num_low_level_actions_per_primitive"] = 10
        # if (
        #     variant["datafile"]
        #     == "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_100_raps_ll_hl_even_rt_drawer-close-v2.hdf5"
        # ):
        #     variant["env_name"] = "drawer-close-v2"
        # elif (
        #     variant["datafile"]
        #     == "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_100_raps_ll_hl_even_rt_soccer-v2.hdf5"
        # ):
        #     variant["env_name"] = "soccer-v2"
        # elif (
        #     variant["datafile"]
        #     == "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_100_raps_ll_hl_even_rt_sweep-into-v2.hdf5"
        # ):
        #     variant["env_name"] = "sweep-into-v2"
        # elif (
        #     variant["datafile"]
        #     == "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_100_raps_ll_hl_even_rt_peg-unplug-side-v2.hdf5"
        # ):
        #     variant["env_name"] = "peg-unplug-side-v2"
        # elif (
        #     variant["datafile"]
        #     == "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_100_raps_ll_hl_even_rt_disassemble-v2.hdf5"
        # ):
        #     variant["env_name"] = "disassemble-v2"
        # elif (
        #     variant["datafile"]
        #     == "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_5_T_25_E_50_P_100_raps_ll_hl_even_rt_assembly-v2.hdf5"
        # ):
        #     variant["env_name"] = "assembly-v2"

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
