import os
from collections import OrderedDict


def visualize_wm(
    env,
    world_model,
    train_actions,
    train_obs,
    test_actions,
    test_obs,
    logdir,
    max_path_length,
    low_level_primitives,
    num_low_level_actions_per_primitive,
    primitive_model=None,
    use_separate_primitives=False,
):
    from rlkit.torch.model_based.dreamer.train_world_model import visualize_rollout

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
        use_separate_primitives=use_separate_primitives,
    )
    visualize_rollout(
        env,
        None,
        None,
        world_model,
        logdir,
        max_path_length,
        use_env=True,
        forcing="teacher",
        tag="none",
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
        primitive_model=primitive_model,
        use_separate_primitives=use_separate_primitives,
    )
    visualize_rollout(
        env,
        None,
        None,
        world_model,
        logdir,
        max_path_length,
        use_env=True,
        forcing="self",
        tag="none",
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
        primitive_model=primitive_model,
        use_separate_primitives=use_separate_primitives,
    )
    visualize_rollout(
        env,
        train_actions,
        train_obs,
        world_model,
        logdir,
        max_path_length,
        use_env=False,
        forcing="none",
        tag="train",
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
        primitive_model=None,
        use_separate_primitives=use_separate_primitives,
    )
    visualize_rollout(
        env,
        train_actions,
        train_obs,
        world_model,
        logdir,
        max_path_length,
        use_env=False,
        forcing="teacher",
        tag="train",
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
        primitive_model=None,
        use_separate_primitives=use_separate_primitives,
    )
    visualize_rollout(
        env,
        train_actions,
        train_obs,
        world_model,
        logdir,
        max_path_length,
        use_env=False,
        forcing="self",
        tag="train",
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
        primitive_model=None,
        use_separate_primitives=use_separate_primitives,
    )
    visualize_rollout(
        env,
        test_actions,
        test_obs,
        world_model,
        logdir,
        max_path_length,
        use_env=False,
        forcing="none",
        tag="test",
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
        primitive_model=None,
        use_separate_primitives=use_separate_primitives,
    )
    visualize_rollout(
        env,
        test_actions,
        test_obs,
        world_model,
        logdir,
        max_path_length,
        use_env=False,
        forcing="teacher",
        tag="test",
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
        primitive_model=None,
        use_separate_primitives=use_separate_primitives,
    )
    visualize_rollout(
        env,
        test_actions,
        test_obs,
        world_model,
        logdir,
        max_path_length,
        use_env=False,
        forcing="self",
        tag="test",
        low_level_primitives=low_level_primitives,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
        primitive_model=None,
        use_separate_primitives=use_separate_primitives,
    )


def experiment(variant):
    import numpy as np
    import torch
    from torch import nn, optim
    from tqdm import tqdm

    import rlkit.torch.pytorch_util as ptu
    from rlkit.core import logger
    from rlkit.envs.primitives_make_env import make_env
    from rlkit.torch.model_based.dreamer.mlp import Mlp, MlpResidual
    from rlkit.torch.model_based.dreamer.train_world_model import (
        compute_world_model_loss,
        get_dataloader,
        get_dataloader_rt,
        get_dataloader_separately,
        update_network,
        visualize_rollout,
        world_model_loss_rt,
    )
    from rlkit.torch.model_based.dreamer.world_models import (
        LowlevelRAPSWorldModel,
        WorldModel,
    )

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
    world_model_loss_kwargs = variant["world_model_loss_kwargs"]
    clone_primitives = variant["clone_primitives"]
    clone_primitives_separately = variant["clone_primitives_separately"]
    clone_primitives_and_train_world_model = variant.get(
        "clone_primitives_and_train_world_model", False
    )
    batch_len = variant.get("batch_len", 100)
    num_epochs = variant["num_epochs"]
    loss_to_use = variant.get("loss_to_use", "both")

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
            env=env,
            **dataloader_kwargs,
        )
    elif clone_primitives_and_train_world_model:
        print("LOADING DATA")
        (
            train_dataloader,
            test_dataloader,
            train_dataset,
            test_dataset,
        ) = get_dataloader_rt(
            variant["datafile"],
            max_path_length=max_path_length * num_low_level_actions_per_primitive + 1,
            **dataloader_kwargs,
        )
    elif low_level_primitives or clone_primitives:
        print("LOADING DATA")
        (
            train_dataloader,
            test_dataloader,
            train_dataset,
            test_dataset,
        ) = get_dataloader(
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

    if clone_primitives_and_train_world_model:
        if variant["mlp_act"] == "elu":
            mlp_act = nn.functional.elu
        elif variant["mlp_act"] == "relu":
            mlp_act = nn.functional.relu
        if variant["mlp_res"]:
            mlp_class = MlpResidual
        else:
            mlp_class = Mlp
        criterion = nn.MSELoss()
        primitive_model = mlp_class(
            hidden_sizes=variant["mlp_hidden_sizes"],
            output_size=low_level_action_dim,
            input_size=250 + env.action_space.low.shape[0] + 1,
            hidden_activation=mlp_act,
        ).to(ptu.device)
        world_model_class = LowlevelRAPSWorldModel
        world_model = world_model_class(
            primitive_model=primitive_model,
            **world_model_kwargs,
        ).to(ptu.device)
        optimizer = optim.Adam(
            world_model.parameters(),
            **optimizer_kwargs,
        )
        best_test_loss = np.inf
        for i in tqdm(range(num_epochs)):
            eval_statistics = OrderedDict()
            print("Epoch: ", i)
            total_primitive_loss = 0
            total_world_model_loss = 0
            total_div_loss = 0
            total_image_pred_loss = 0
            total_transition_loss = 0
            total_entropy_loss = 0
            total_pred_discount_loss = 0
            total_reward_pred_loss = 0
            total_train_steps = 0
            for data in train_dataloader:
                with torch.cuda.amp.autocast():
                    (
                        high_level_actions,
                        obs,
                        rewards,
                        terminals,
                    ), low_level_actions = data
                    obs = obs.to(ptu.device).float()
                    low_level_actions = low_level_actions.to(ptu.device).float()
                    high_level_actions = high_level_actions.to(ptu.device).float()
                    rewards = rewards.to(ptu.device).float()
                    terminals = terminals.to(ptu.device).float()
                    assert all(terminals[:, -1] == 1)
                    rt_idxs = np.arange(
                        num_low_level_actions_per_primitive,
                        obs.shape[1],
                        num_low_level_actions_per_primitive,
                    )
                    rt_idxs = np.concatenate(
                        [[0], rt_idxs]
                    )  # reset obs, effect of first primitive, second primitive, so on

                    batch_start = np.random.randint(
                        0, obs.shape[1] - batch_len, size=(obs.shape[0])
                    )
                    batch_indices = np.linspace(
                        batch_start,
                        batch_start + batch_len,
                        batch_len,
                        endpoint=False,
                    ).astype(int)
                    (
                        post,
                        prior,
                        post_dist,
                        prior_dist,
                        image_dist,
                        reward_dist,
                        pred_discount_dist,
                        _,
                        action_preds,
                    ) = world_model(
                        obs,
                        (high_level_actions, low_level_actions),
                        use_network_action=False,
                        batch_indices=batch_indices,
                        rt_idxs=rt_idxs,
                    )
                    obs = world_model.flatten_obs(
                        obs[np.arange(batch_indices.shape[1]), batch_indices].permute(
                            1, 0, 2
                        ),
                        (int(np.prod(image_shape)),),
                    )
                    rewards = rewards.reshape(-1, rewards.shape[-1])
                    terminals = terminals.reshape(-1, terminals.shape[-1])
                    (
                        world_model_loss,
                        div,
                        image_pred_loss,
                        reward_pred_loss,
                        transition_loss,
                        entropy_loss,
                        pred_discount_loss,
                    ) = world_model_loss_rt(
                        world_model,
                        image_shape,
                        image_dist,
                        reward_dist,
                        {
                            key: value[np.arange(batch_indices.shape[1]), batch_indices]
                            .permute(1, 0, 2)
                            .reshape(-1, value.shape[-1])
                            for key, value in prior.items()
                        },
                        {
                            key: value[np.arange(batch_indices.shape[1]), batch_indices]
                            .permute(1, 0, 2)
                            .reshape(-1, value.shape[-1])
                            for key, value in post.items()
                        },
                        prior_dist,
                        post_dist,
                        pred_discount_dist,
                        obs,
                        rewards,
                        terminals,
                        **world_model_loss_kwargs,
                    )

                    batch_start = np.random.randint(
                        0,
                        low_level_actions.shape[1] - batch_len,
                        size=(low_level_actions.shape[0]),
                    )
                    batch_indices = np.linspace(
                        batch_start,
                        batch_start + batch_len,
                        batch_len,
                        endpoint=False,
                    ).astype(int)
                    primitive_loss = criterion(
                        action_preds[np.arange(batch_indices.shape[1]), batch_indices]
                        .permute(1, 0, 2)
                        .reshape(-1, action_preds.shape[-1]),
                        low_level_actions[:, 1:][
                            np.arange(batch_indices.shape[1]), batch_indices
                        ]
                        .permute(1, 0, 2)
                        .reshape(-1, action_preds.shape[-1]),
                    )
                    total_primitive_loss += primitive_loss.item()
                    total_world_model_loss += world_model_loss.item()
                    total_div_loss += div.item()
                    total_image_pred_loss += image_pred_loss.item()
                    total_transition_loss += transition_loss.item()
                    total_entropy_loss += entropy_loss.item()
                    total_pred_discount_loss += pred_discount_loss.item()
                    total_reward_pred_loss += reward_pred_loss.item()

                    if loss_to_use == "wm":
                        loss = world_model_loss
                    elif loss_to_use == "primitive":
                        loss = primitive_loss
                    else:
                        loss = world_model_loss + primitive_loss
                    total_train_steps += 1

                update_network(world_model, optimizer, loss, gradient_clip, scaler)
                scaler.update()
            eval_statistics["train/primitive_loss"] = (
                total_primitive_loss / total_train_steps
            )
            eval_statistics["train/world_model_loss"] = (
                total_world_model_loss / total_train_steps
            )
            eval_statistics["train/image_pred_loss"] = (
                total_image_pred_loss / total_train_steps
            )
            eval_statistics["train/transition_loss"] = (
                total_transition_loss / total_train_steps
            )
            eval_statistics["train/entropy_loss"] = (
                total_entropy_loss / total_train_steps
            )
            eval_statistics["train/pred_discount_loss"] = (
                total_pred_discount_loss / total_train_steps
            )
            eval_statistics["train/reward_pred_loss"] = (
                total_reward_pred_loss / total_train_steps
            )
            latest_state_dict = world_model.state_dict().copy()
            with torch.no_grad():
                total_primitive_loss = 0
                total_world_model_loss = 0
                total_div_loss = 0
                total_image_pred_loss = 0
                total_transition_loss = 0
                total_entropy_loss = 0
                total_pred_discount_loss = 0
                total_reward_pred_loss = 0
                total_loss = 0
                total_test_steps = 0
                for data in test_dataloader:
                    with torch.cuda.amp.autocast():
                        (
                            high_level_actions,
                            obs,
                            rewards,
                            terminals,
                        ), low_level_actions = data
                        obs = obs.to(ptu.device).float()
                        low_level_actions = low_level_actions.to(ptu.device).float()
                        high_level_actions = high_level_actions.to(ptu.device).float()
                        rewards = rewards.to(ptu.device).float()
                        terminals = terminals.to(ptu.device).float()
                        assert all(terminals[:, -1] == 1)
                        rt_idxs = np.arange(
                            num_low_level_actions_per_primitive,
                            obs.shape[1],
                            num_low_level_actions_per_primitive,
                        )
                        rt_idxs = np.concatenate(
                            [[0], rt_idxs]
                        )  # reset obs, effect of first primitive, second primitive, so on

                        batch_start = np.random.randint(
                            0, obs.shape[1] - batch_len, size=(obs.shape[0])
                        )
                        batch_indices = np.linspace(
                            batch_start,
                            batch_start + batch_len,
                            batch_len,
                            endpoint=False,
                        ).astype(int)
                        (
                            post,
                            prior,
                            post_dist,
                            prior_dist,
                            image_dist,
                            reward_dist,
                            pred_discount_dist,
                            _,
                            action_preds,
                        ) = world_model(
                            obs,
                            (high_level_actions, low_level_actions),
                            use_network_action=False,
                            batch_indices=batch_indices,
                            rt_idxs=rt_idxs,
                        )
                        obs = world_model.flatten_obs(
                            obs[
                                np.arange(batch_indices.shape[1]), batch_indices
                            ].permute(1, 0, 2),
                            (int(np.prod(image_shape)),),
                        )
                        rewards = rewards.reshape(-1, rewards.shape[-1])
                        terminals = terminals.reshape(-1, terminals.shape[-1])
                        (
                            world_model_loss,
                            div,
                            image_pred_loss,
                            reward_pred_loss,
                            transition_loss,
                            entropy_loss,
                            pred_discount_loss,
                        ) = world_model_loss_rt(
                            world_model,
                            image_shape,
                            image_dist,
                            reward_dist,
                            {
                                key: value[
                                    np.arange(batch_indices.shape[1]), batch_indices
                                ]
                                .permute(1, 0, 2)
                                .reshape(-1, value.shape[-1])
                                for key, value in prior.items()
                            },
                            {
                                key: value[
                                    np.arange(batch_indices.shape[1]), batch_indices
                                ]
                                .permute(1, 0, 2)
                                .reshape(-1, value.shape[-1])
                                for key, value in post.items()
                            },
                            prior_dist,
                            post_dist,
                            pred_discount_dist,
                            obs,
                            rewards,
                            terminals,
                            **world_model_loss_kwargs,
                        )

                        batch_start = np.random.randint(
                            0,
                            low_level_actions.shape[1] - batch_len,
                            size=(low_level_actions.shape[0]),
                        )
                        batch_indices = np.linspace(
                            batch_start,
                            batch_start + batch_len,
                            batch_len,
                            endpoint=False,
                        ).astype(int)
                        primitive_loss = criterion(
                            action_preds[
                                np.arange(batch_indices.shape[1]), batch_indices
                            ]
                            .permute(1, 0, 2)
                            .reshape(-1, action_preds.shape[-1]),
                            low_level_actions[:, 1:][
                                np.arange(batch_indices.shape[1]), batch_indices
                            ]
                            .permute(1, 0, 2)
                            .reshape(-1, action_preds.shape[-1]),
                        )
                        total_primitive_loss += primitive_loss.item()
                        total_world_model_loss += world_model_loss.item()
                        total_div_loss += div.item()
                        total_image_pred_loss += image_pred_loss.item()
                        total_transition_loss += transition_loss.item()
                        total_entropy_loss += entropy_loss.item()
                        total_pred_discount_loss += pred_discount_loss.item()
                        total_reward_pred_loss += reward_pred_loss.item()
                        total_loss += world_model_loss.item() + primitive_loss.item()
                        total_test_steps += 1
                eval_statistics["test/primitive_loss"] = (
                    total_primitive_loss / total_test_steps
                )
                eval_statistics["test/world_model_loss"] = (
                    total_world_model_loss / total_test_steps
                )
                eval_statistics["test/image_pred_loss"] = (
                    total_image_pred_loss / total_test_steps
                )
                eval_statistics["test/transition_loss"] = (
                    total_transition_loss / total_test_steps
                )
                eval_statistics["test/entropy_loss"] = (
                    total_entropy_loss / total_test_steps
                )
                eval_statistics["test/pred_discount_loss"] = (
                    total_pred_discount_loss / total_test_steps
                )
                eval_statistics["test/reward_pred_loss"] = (
                    total_reward_pred_loss / total_test_steps
                )
                if (total_loss / total_test_steps) <= best_test_loss:
                    best_test_loss = total_loss / total_test_steps
                    os.makedirs(logdir + "/models/", exist_ok=True)
                    best_wm_state_dict = world_model.state_dict().copy()
                    torch.save(
                        best_wm_state_dict,
                        logdir + "/models/world_model.pt",
                    )
                if i % variant["plotting_period"] == 0:
                    print("Best test loss", best_test_loss)
                    world_model.load_state_dict(best_wm_state_dict)
                    visualize_wm(
                        env,
                        world_model,
                        train_dataset.outputs,
                        train_dataset.inputs[1],
                        test_dataset.outputs,
                        test_dataset.inputs[1],
                        logdir,
                        max_path_length,
                        low_level_primitives,
                        num_low_level_actions_per_primitive,
                        primitive_model=primitive_model,
                    )
                    world_model.load_state_dict(latest_state_dict)
                logger.record_dict(eval_statistics, prefix="")
                logger.dump_tabular(with_prefix=False, with_timestamp=False)

    elif clone_primitives_separately:
        world_model.load_state_dict(torch.load(variant["world_model_path"]))
        criterion = nn.MSELoss()
        primitives = []
        for p in range(env.num_primitives):
            arguments_size = train_datasets[p].inputs[0].shape[-1]
            m = Mlp(
                hidden_sizes=variant["mlp_hidden_sizes"],
                output_size=low_level_action_dim,
                input_size=world_model.feature_size + arguments_size,
                hidden_activation=torch.nn.functional.relu,
            ).to(ptu.device)
            if variant.get("primitives_path", None):
                m.load_state_dict(
                    torch.load(
                        variant["primitives_path"] + "primitive_model_{}.pt".format(p)
                    )
                )
            primitives.append(m)

        optimizers = [
            optim.Adam(p.parameters(), **optimizer_kwargs) for p in primitives
        ]
        for i in tqdm(range(num_epochs)):
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
                visualize_rollout(
                    env,
                    None,
                    None,
                    world_model,
                    logdir,
                    max_path_length,
                    use_env=True,
                    forcing="teacher",
                    tag="none",
                    low_level_primitives=low_level_primitives,
                    num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
                    primitive_model=primitives,
                    use_separate_primitives=True,
                )
                visualize_rollout(
                    env,
                    None,
                    None,
                    world_model,
                    logdir,
                    max_path_length,
                    use_env=True,
                    forcing="self",
                    tag="none",
                    low_level_primitives=low_level_primitives,
                    num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
                    primitive_model=primitives,
                    use_separate_primitives=True,
                )
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
                            primitive_model.state_dict(),
                            logdir + "/models/primitive_model_{}.pt".format(p),
                        )
            logger.record_dict(eval_statistics, prefix="")
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
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
        world_model.load_state_dict(torch.load(variant["world_model_path"]))
        criterion = nn.MSELoss()
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
        for i in tqdm(range(num_epochs)):
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
                        primitive_model.state_dict(),
                        logdir + "/models/primitive_model.pt",
                    )
                logger.record_dict(eval_statistics, prefix="")
                logger.dump_tabular(with_prefix=False, with_timestamp=False)
    else:
        world_model = WorldModel(**world_model_kwargs).to(ptu.device)
        optimizer = optim.Adam(
            world_model.parameters(),
            **optimizer_kwargs,
        )
        for i in tqdm(range(num_epochs)):
            if i % variant["plotting_period"] == 0:
                visualize_wm(
                    env,
                    world_model,
                    train_dataset.inputs,
                    train_dataset.outputs,
                    test_dataset.inputs,
                    test_dataset.outputs,
                    logdir,
                    max_path_length,
                    low_level_primitives,
                    num_low_level_actions_per_primitive,
                )
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
                        obs.permute(1, 0, 2), (int(np.prod(image_shape)),)
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
                            obs.permute(1, 0, 2), (int(np.prod(image_shape)),)
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

        world_model.load_state_dict(torch.load(logdir + "/models/world_model.pt"))
        visualize_wm(
            env,
            world_model,
            train_dataset,
            test_dataset,
            logdir,
            max_path_length,
            low_level_primitives,
            num_low_level_actions_per_primitive,
        )
