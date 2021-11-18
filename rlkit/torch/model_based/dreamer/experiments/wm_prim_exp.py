import os
from collections import OrderedDict

import numpy as np

from rlkit.torch.model_based.dreamer.train_world_model import get_dataloader_separately


def visualize_wm(
    env,
    world_model,
    train_dataset,
    test_dataset,
    logdir,
    max_path_length,
    i,
    low_level_primitives,
    num_low_level_actions_per_primitive,
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
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
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
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
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
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
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
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
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
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
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
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
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
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
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
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
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
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
    )


def experiment(variant):
    import torch
    from torch import nn, optim
    from tqdm import tqdm

    import rlkit.torch.pytorch_util as ptu
    from rlkit.core import logger
    from rlkit.envs.primitives_make_env import make_env
    from rlkit.torch.model_based.dreamer.mlp import Mlp
    from rlkit.torch.model_based.dreamer.train_world_model import (
        compute_world_model_loss,
        get_dataloader,
        update_network,
        visualize_rollout,
    )
    from rlkit.torch.model_based.dreamer.world_models import WorldModel

    env_suite, env_name, env_kwargs = (
        variant["env_suite"],
        variant["env_name"],
        variant["env_kwargs"],
    )
    max_path_length = variant["env_kwargs"]["max_path_length"]
    low_level_primitives = variant["low_level_primitives"]
    num_low_level_actions_per_primitive = variant["num_low_level_actions_per_primitive"]
    low_level_action_dim = variant["low_level_action_dim"]
    dataloader_kwargs = variant["dataloader_kwargs"]
    env = make_env(env_suite, env_name, env_kwargs)
    world_model_kwargs = variant["model_kwargs"]
    optimizer_kwargs = variant["optimizer_kwargs"]
    gradient_clip = variant["gradient_clip"]
    if low_level_primitives:
        world_model_kwargs["action_dim"] = low_level_action_dim
    else:
        world_model_kwargs["action_dim"] = env.action_space.low.shape[0]
    image_shape = env.image_shape
    world_model_kwargs["image_shape"] = image_shape
    scaler = torch.cuda.amp.GradScaler()
    world_model = WorldModel(
        **world_model_kwargs,
    ).to(ptu.device)
    world_model_loss_kwargs = variant["world_model_loss_kwargs"]
    clone_primitives = variant["clone_primitives"]
    clone_primitives_separately = variant["clone_primitives_separately"]
    num_epochs = variant["num_epochs"]

    logdir = logger.get_snapshot_dir()

    if clone_primitives_separately:
        (
            train_dataloaders,
            test_dataloaders,
            train_datasets,
            test_datasets,
        ) = get_dataloader_separately(
            variant["datafile"],
            num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
            num_primitives=env.num_primitives,
            **dataloader_kwargs,
        )
    elif low_level_primitives or clone_primitives:
        train_dataloader, test_dataloader, train_dataset, test_dataset = get_dataloader(
            variant["datafile"],
            max_path_length=max_path_length * num_low_level_actions_per_primitive + 1,
            **dataloader_kwargs,
        )
    else:
        train_dataloader, test_dataloader, train_dataset, test_dataset = get_dataloader(
            variant["datafile"],
            max_path_length=max_path_length + 1,
            **dataloader_kwargs,
        )

    if variant["visualize_wm_from_path"]:
        world_model.load_state_dict(torch.load(variant["world_model_path"]))
        visualize_wm(
            env,
            world_model,
            train_dataset,
            test_dataset,
            logdir,
            max_path_length,
            -1,
            low_level_primitives,
            num_low_level_actions_per_primitive,
        )
    elif clone_primitives_separately:
        world_model.load_state_dict(torch.load(variant["world_model_path"]))
        criterion = nn.MSELoss()
        primitives = []
        for i in range(env.num_primitives):
            arguments_size = train_datasets[i].inputs[0].shape[-1]
            primitives.append(
                Mlp(
                    hidden_sizes=variant["mlp_hidden_sizes"],
                    output_size=low_level_action_dim,
                    input_size=world_model.feature_size + arguments_size,
                    hidden_activation=torch.nn.functional.relu,
                ).to(ptu.device)
            )
        optimizers = [
            optim.Adam(p.parameters(), **optimizer_kwargs) for p in primitives
        ]
        for i in tqdm(range(num_epochs)):
            eval_statistics = OrderedDict()
            print("Epoch: ", i)
            for p, (
                train_dataloader,
                test_dataloader,
                primitive_model,
                optimizer,
            ) in enumerate(
                zip(train_dataloaders, test_dataloaders, primitives, optimizers)
            ):
                total_loss = 0
                total_train_steps = 0
                for data in train_dataloader:
                    with torch.cuda.amp.autocast():
                        (arguments, obs), actions = data
                        obs = obs.to(ptu.device).float()
                        actions = actions.to(ptu.device).float()
                        arguments = arguments.to(ptu.device).float()
                        action_preds = world_model(
                            obs,
                            (arguments, actions),
                            primitive_model,
                            use_network_action=False,
                        )[-1]
                        loss = criterion(action_preds, actions)
                        total_loss += loss.item()
                        total_train_steps += 1

                    update_network(
                        primitive_model, optimizer, loss, gradient_clip, scaler
                    )
                    scaler.update()
                eval_statistics["train/primitive_loss {}".format(p)] = (
                    total_loss / total_train_steps
                )
                best_test_loss = np.inf
                with torch.no_grad():
                    total_loss = 0
                    total_test_steps = 0
                    for data in test_dataloader:
                        with torch.cuda.amp.autocast():
                            (high_level_actions, obs), actions = data
                            obs = obs.to(ptu.device).float()
                            actions = actions.to(ptu.device).float()
                            high_level_actions = high_level_actions.to(
                                ptu.device
                            ).float()
                            action_preds = world_model(
                                obs,
                                (high_level_actions, actions),
                                primitive_model,
                                use_network_action=False,
                            )[-1]
                            loss = criterion(action_preds, actions)
                            total_loss += loss.item()
                            total_test_steps += 1
                    eval_statistics["test/primitive_loss {}".format(p)] = (
                        total_loss / total_test_steps
                    )
                    if (total_loss / total_test_steps) <= best_test_loss:
                        best_test_loss = total_loss / total_test_steps
                        os.makedirs(logdir + "/models/", exist_ok=True)
                        torch.save(
                            world_model.state_dict(),
                            logdir + "/models/primitive_model_{}.pt".format(p),
                        )
            logger.record_dict(eval_statistics, prefix="")
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
            if i % variant["plotting_period"] == 0:
                visualize_rollout(
                    env,
                    None,
                    None,
                    world_model,
                    logdir,
                    max_path_length,
                    use_env=True,
                    forcing="none",
                    tag="none",
                    low_level_primitives=low_level_primitives,
                    num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
                    primitive_model=primitives,
                    use_separate_primitives=True,
                )

    elif clone_primitives:
        primitive_model = Mlp(
            hidden_sizes=variant["mlp_hidden_sizes"],
            output_size=low_level_action_dim,
            input_size=world_model.feature_size + env.action_space.low.shape[0] + 1,
            hidden_activation=torch.nn.functional.relu,
        ).to(ptu.device)
        optimizer = optim.Adam(
            primitive_model.parameters(),
            **optimizer_kwargs,
        )
        world_model.load_state_dict(torch.load(variant["world_model_path"]))
        criterion = nn.MSELoss()
        for i in tqdm(range(num_epochs)):
            eval_statistics = OrderedDict()
            print("Epoch: ", i)
            total_loss = 0
            total_train_steps = 0
            for data in train_dataloader:
                with torch.cuda.amp.autocast():
                    (high_level_actions, obs), actions = data
                    obs = obs.to(ptu.device).float()
                    actions = actions.to(ptu.device).float()
                    high_level_actions = high_level_actions.to(ptu.device).float()
                    action_preds = world_model(
                        obs,
                        (high_level_actions, actions),
                        primitive_model,
                        use_network_action=False,
                    )[-1]
                    loss = criterion(action_preds, actions)
                    total_loss += loss.item()
                    total_train_steps += 1

                update_network(primitive_model, optimizer, loss, gradient_clip, scaler)
                scaler.update()
            eval_statistics["train/primitive_loss"] = total_loss / total_train_steps
            best_test_loss = np.inf
            with torch.no_grad():
                total_loss = 0
                total_test_steps = 0
                for data in test_dataloader:
                    with torch.cuda.amp.autocast():
                        (high_level_actions, obs), actions = data
                        obs = obs.to(ptu.device).float()
                        actions = actions.to(ptu.device).float()
                        high_level_actions = high_level_actions.to(ptu.device).float()
                        action_preds = world_model(
                            obs,
                            (high_level_actions, actions),
                            primitive_model,
                            use_network_action=False,
                        )[-1]
                        loss = criterion(action_preds, actions)
                        total_loss += loss.item()
                        total_test_steps += 1
                eval_statistics["test/primitive_loss"] = total_loss / total_test_steps
                if (total_loss / total_test_steps) <= best_test_loss:
                    best_test_loss = total_loss / total_test_steps
                    os.makedirs(logdir + "/models/", exist_ok=True)
                    torch.save(
                        world_model.state_dict(),
                        logdir + "/models/primitive_model.pt",
                    )
                logger.record_dict(eval_statistics, prefix="")
                logger.dump_tabular(with_prefix=False, with_timestamp=False)
                if i % variant["plotting_period"] == 0:
                    visualize_rollout(
                        env,
                        None,
                        None,
                        world_model,
                        logdir,
                        max_path_length,
                        use_env=True,
                        forcing="none",
                        tag="none",
                        low_level_primitives=low_level_primitives,
                        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
                        primitive_model=primitive_model,
                    )
                    visualize_rollout(
                        env,
                        train_dataset.outputs,
                        train_dataset.inputs[1],
                        world_model,
                        logdir,
                        max_path_length,
                        use_env=False,
                        forcing="teacher",
                        tag="train",
                        low_level_primitives=low_level_primitives,
                        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive
                        - 1,
                    )
                    visualize_rollout(
                        env,
                        test_dataset.outputs,
                        test_dataset.inputs[1],
                        world_model,
                        logdir,
                        max_path_length,
                        use_env=False,
                        forcing="teacher",
                        tag="test",
                        low_level_primitives=low_level_primitives,
                        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive
                        - 1,
                    )
    else:
        optimizer = optim.Adam(
            world_model.parameters(),
            **optimizer_kwargs,
        )
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
                    actions, obs = data
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
                        actions, obs = data
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
                    low_level_primitives,
                    num_low_level_actions_per_primitive,
                )

        world_model.load_state_dict(torch.load(logdir + "/models/world_model.pt"))
        visualize_wm(
            env,
            world_model,
            train_dataset,
            test_dataset,
            logdir,
            max_path_length,
            -1,
            low_level_primitives,
            num_low_level_actions_per_primitive,
        )
