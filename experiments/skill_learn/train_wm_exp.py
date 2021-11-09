import argparse
import os
import random
import subprocess
from collections import OrderedDict

import numpy as np

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment


def visualize_wm(
    env, world_model, train_dataset, test_dataset, logdir, max_path_length, i
):
    from rlkit.torch.model_based.dreamer.train_world_model import visualize_rollout

    visualize_rollout(
        env,
        None,
        world_model,
        logdir,
        max_path_length,
        i,
        use_env=True,
        forcing="none",
        tag="none",
    )
    visualize_rollout(
        env,
        None,
        world_model,
        logdir,
        max_path_length,
        i,
        use_env=True,
        forcing="teacher",
        tag="none",
    )
    visualize_rollout(
        env,
        None,
        world_model,
        logdir,
        max_path_length,
        i,
        use_env=True,
        forcing="self",
        tag="none",
    )

    visualize_rollout(
        env,
        train_dataset,
        world_model,
        logdir,
        max_path_length,
        i,
        use_env=False,
        forcing="none",
        tag="train",
    )
    visualize_rollout(
        env,
        train_dataset,
        world_model,
        logdir,
        max_path_length,
        i,
        use_env=False,
        forcing="teacher",
        tag="train",
    )
    visualize_rollout(
        env,
        train_dataset,
        world_model,
        logdir,
        max_path_length,
        i,
        use_env=False,
        forcing="self",
        tag="train",
    )
    visualize_rollout(
        env,
        test_dataset,
        world_model,
        logdir,
        max_path_length,
        i,
        use_env=False,
        forcing="none",
        tag="test",
    )
    visualize_rollout(
        env,
        test_dataset,
        world_model,
        logdir,
        max_path_length,
        i,
        use_env=False,
        forcing="teacher",
        tag="test",
    )
    visualize_rollout(
        env,
        test_dataset,
        world_model,
        logdir,
        max_path_length,
        i,
        use_env=False,
        forcing="self",
        tag="test",
    )


def experiment(variant):
    import torch
    from torch import optim
    from tqdm import tqdm

    import rlkit.torch.pytorch_util as ptu
    from rlkit.core import logger
    from rlkit.envs.primitives_make_env import make_env
    from rlkit.torch.model_based.dreamer.train_world_model import (
        compute_world_model_loss,
        get_dataloader,
        update_network,
    )
    from rlkit.torch.model_based.dreamer.world_models import WorldModel

    env_suite, env_name, env_kwargs = (
        variant["env_suite"],
        variant["env_name"],
        variant["env_kwargs"],
    )
    max_path_length = variant["env_kwargs"]["max_path_length"]
    batch_len = variant["batch_len"]
    batch_size = variant["batch_size"]
    train_test_split = variant["train_test_split"]
    env = make_env(env_suite, env_name, env_kwargs)
    world_model_kwargs = variant["model_kwargs"]
    optimizer_kwargs = variant["optimizer_kwargs"]
    gradient_clip = variant["gradient_clip"]
    world_model_kwargs["action_dim"] = env.action_space.low.shape[0]
    image_shape = env.image_shape
    world_model_kwargs["image_shape"] = image_shape
    scaler = torch.cuda.amp.GradScaler()
    world_model = WorldModel(
        **world_model_kwargs,
    ).to(ptu.device)
    world_model_loss_kwargs = variant["world_model_loss_kwargs"]

    num_epochs = variant["num_epochs"]
    optimizer = optim.Adam(
        world_model.parameters(),
        **optimizer_kwargs,
    )
    logdir = logger.get_snapshot_dir()

    train_dataloader, test_dataloader, train_dataset, test_dataset = get_dataloader(
        variant["datafile"], train_test_split, batch_len, batch_size, max_path_length
    )

    if variant["visualize_wm_from_path"]:
        world_model.load_state_dict(torch.load(variant["world_model_path"]))
        visualize_wm(
            env, world_model, train_dataset, test_dataset, logdir, max_path_length, -1
        )
    else:
        for i in tqdm(range(num_epochs)):
            eval_statistics = OrderedDict()
            print("Epoch: ", i)
            total_wm_loss = 0
            total_div_loss = 0
            total_image_pred_loss = 0
            total_transition_loss = 0
            total_entropy_loss = 0
            total_train_steps = 0
            for data in train_dataloader:
                with torch.cuda.amp.autocast():
                    obs, actions = data
                    obs = obs.to(ptu.device).float()
                    actions = actions.to(ptu.device).float()
                    post, prior, post_dist, prior_dist, image_dist = world_model(
                        obs, actions
                    )[:5]
                    obs = world_model.flatten_obs(
                        obs.transpose(1, 0), (int(np.prod(image_shape)),)
                    )
                    (
                        world_model_loss,
                        div,
                        image_pred_loss,
                        transition_loss,
                        entropy_loss,
                    ) = compute_world_model_loss(
                        world_model,
                        image_shape,
                        image_dist,
                        prior,
                        post,
                        prior_dist,
                        post_dist,
                        obs,
                        **world_model_loss_kwargs,
                    )
                    total_wm_loss += world_model_loss.item()
                    total_div_loss += div.item()
                    total_image_pred_loss += image_pred_loss.item()
                    total_transition_loss += transition_loss.item()
                    total_entropy_loss += entropy_loss.item()
                    total_train_steps += 1

                update_network(
                    world_model, optimizer, world_model_loss, gradient_clip, scaler
                )
                scaler.update()
            eval_statistics["train/wm_loss"] = total_wm_loss / total_train_steps
            eval_statistics["train/div_loss"] = total_div_loss / total_train_steps
            eval_statistics["train/image_pred_loss"] = (
                total_image_pred_loss / total_train_steps
            )
            eval_statistics["train/transition_loss"] = (
                total_transition_loss / total_train_steps
            )
            eval_statistics["train/entropy_loss"] = (
                total_entropy_loss / total_train_steps
            )
            best_test_loss = np.inf
            with torch.no_grad():
                total_wm_loss = 0
                total_div_loss = 0
                total_image_pred_loss = 0
                total_transition_loss = 0
                total_entropy_loss = 0
                total_train_steps = 0
                total_test_steps = 0
                for data in test_dataloader:
                    with torch.cuda.amp.autocast():
                        obs, actions = data
                        obs = obs.to(ptu.device).float()
                        actions = actions.to(ptu.device).float()
                        post, prior, post_dist, prior_dist, image_dist = world_model(
                            obs, actions
                        )[:5]
                        obs = world_model.flatten_obs(
                            obs.transpose(1, 0), (int(np.prod(image_shape)),)
                        )
                        (
                            world_model_loss,
                            div,
                            image_pred_loss,
                            transition_loss,
                            entropy_loss,
                        ) = compute_world_model_loss(
                            world_model,
                            image_shape,
                            image_dist,
                            prior,
                            post,
                            prior_dist,
                            post_dist,
                            obs,
                            **world_model_loss_kwargs,
                        )
                        total_wm_loss += world_model_loss.item()
                        total_div_loss += div.item()
                        total_image_pred_loss += image_pred_loss.item()
                        total_transition_loss += transition_loss.item()
                        total_entropy_loss += entropy_loss.item()
                        total_test_steps += 1
                eval_statistics["test/wm_loss"] = total_wm_loss / total_test_steps
                eval_statistics["test/div_loss"] = total_div_loss / total_test_steps
                eval_statistics["test/image_pred_loss"] = (
                    total_image_pred_loss / total_test_steps
                )
                eval_statistics["test/transition_loss"] = (
                    total_transition_loss / total_test_steps
                )
                eval_statistics["test/entropy_loss"] = (
                    total_entropy_loss / total_test_steps
                )
                if (total_wm_loss / total_test_steps) <= best_test_loss:
                    best_test_loss = total_wm_loss / total_test_steps
                    os.makedirs(logdir + "/models/", exist_ok=True)
                    torch.save(
                        world_model.state_dict(),
                        logdir + "/models/world_model.pt",
                    )
                logger.record_dict(eval_statistics, prefix="")
                logger.dump_tabular(with_prefix=False, with_timestamp=False)
            if i % variant["plotting_period"] == 0:
                visualize_wm(
                    env,
                    world_model,
                    train_dataset,
                    test_dataset,
                    logdir,
                    max_path_length,
                    i,
                )

        world_model.load_state_dict(torch.load(logdir + "/models/world_model.pt"))
        visualize_wm(
            env, world_model, train_dataset, test_dataset, logdir, max_path_length, -1
        )


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
        plotting_period=25,
        env_kwargs=dict(
            control_mode="end_effector",
            action_scale=1,
            max_path_length=50,
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
                max_path_length=50,
                unflatten_images=False,
            ),
            image_kwargs=dict(imwidth=64, imheight=64),
        ),
        env_suite="metaworld",
        env_name="reach-v2",
        world_model_loss_kwargs=dict(
            forward_kl=False,
            free_nats=1.0,
            transition_loss_scale=0.8,
            kl_loss_scale=0.0,
            image_loss_scale=1.0,
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
            lr=3e-4,
            eps=1e-5,
            weight_decay=0.0,
        ),
        gradient_clip=100,
        batch_len=50,
        batch_size=50,
        num_epochs=1000,
        datafile="/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_50_T_250_E_50_ll.hdf5",
        train_test_split=0.8,
    )

    search_space = {
        # 'batch_len':[50, 100, 250, 500],
        # 'batch_size':[50, 100, 250, 500],
        # 'datafile':["/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_500_T_25_E_50_ll.hdf5"],
        "datafile": [
            "/home/mdalal/research/skill_learn/rlkit/data/world_model_data/wm_H_50_T_250_E_50_ll.hdf5"
        ],
        "env_kwargs.max_path_length": [500],
        "env_kwargs.max_path_length": [50],
        "env_kwargs.usage_kwargs.max_path_length": [50],
        # 'env_kwargs.usage_kwargs.max_path_length':[500],
        "env_kwargs.control_mode": ["end_effector"],
        "num_epochs": [1000],
        "model_kwargs.use_prior_instead_of_posterior": [True, False],
        "plotting_period": [50, 100],
        "visualize_wm_from_path": [True],
        "world_model_path": [
            "/home/mdalal/research/skill_learn/rlkit/data/11-09-ll-prior-v1/11-09-ll_prior_v1_2021_11_09_12_30_46_0000--s-45920/models/world_model.pt"
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
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
