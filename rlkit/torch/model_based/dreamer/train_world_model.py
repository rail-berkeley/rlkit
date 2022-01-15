import os
import os.path as osp

import cv2
import h5py
import matplotlib
import numpy as np
import torch
from torch.distributions import kl_divergence as kld
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, Sampler

import rlkit.torch.pytorch_util as ptu

matplotlib.use("Agg")


class BatchLenRandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, data_source, num_samples=None, generator=None):
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator
        self.max_path_length = data_source.max_path_length
        self.batch_len = data_source.batch_len

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item())
            )
        else:
            generator = self.generator
        possible_batch_indices = [
            np.linspace(
                batch_start,
                batch_start + self.batch_len,
                self.batch_len,
                endpoint=False,
            ).astype(int)
            for batch_start in range(0, self.max_path_length - self.batch_len)
        ]
        possible_batch_indices = np.random.permutation(possible_batch_indices)
        random_batches = torch.randperm(n, generator=generator).tolist()
        total_list = []
        for batch_idx in possible_batch_indices:
            for random_batch in random_batches:
                total_list.append([random_batch, batch_idx])
        yield from total_list

    def __len__(self):
        return self.num_samples


def subsample_paths(actions, observations, num_inputs, num_outputs):
    # idxs = np.linspace(0, num_inputs, num_outputs + 1)
    # spacing = num_inputs // (num_outputs)
    # a = actions.reshape(num_outputs, spacing, -1)
    # a = a.sum(axis=1)[:, :3]  # just keep sum of xyz deltas
    # actions = np.concatenate((a, actions[idxs.astype(np.int)[1:] - 1, 3:]), axis=1)
    # observations = observations[idxs.astype(np.int)[1:] - 1]
    return actions, observations


def get_state(o, a, state, world_model, forcing, new_img, first_output=False):
    if forcing == "teacher" or first_output:
        o = ptu.from_numpy(o.reshape(1, -1))
        embed = world_model.encode(o)
        state, prior = world_model.obs_step(state, a, embed)
    elif forcing == "self" and world_model.use_prior_instead_of_posterior:
        new_img = new_img.reshape(-1, o.shape[-1])
        new_img = torch.clamp(new_img + 0.5, 0, 1) * 255.0
        embed = world_model.encode(new_img)
        state, prior = world_model.obs_step(state, a, embed)
    else:
        state = world_model.action_step(state, a)
        prior = state
    return state, prior


def forward_low_level_primitive(
    ll_a,
    ll_o,
    world_model,
    num_low_level_actions_per_primitive,
    primitive_model,
    net,
    forcing,
    new_img,
    high_level,
    state,
    primitive_name=None,
):
    reconstructions = []
    total_err = 0
    for k in range(0, num_low_level_actions_per_primitive):
        a = ptu.from_numpy(ll_a[k : k + 1])
        o = ll_o[k]
        if primitive_model:
            tmp = np.array([(k + 1) / (num_low_level_actions_per_primitive)]).reshape(
                1, -1
            )
            hl = np.concatenate((high_level, tmp), 1)
            inp = torch.cat(
                [ptu.from_numpy(hl), world_model.get_features(state)], dim=1
            )
            a_pred = net(inp)
            total_err += torch.nn.functional.mse_loss(a_pred, a).item()
            a = a_pred

        state, prior = get_state(o, a, state, world_model, forcing, new_img)
        if world_model.use_prior_instead_of_posterior:
            new_img = world_model.decode(world_model.get_features(prior))
        else:
            new_img = world_model.decode(world_model.get_features(state))
        reconstructions.append(new_img.unsqueeze(1))
    if primitive_name is not None:
        print(primitive_name, total_err / num_low_level_actions_per_primitive)
    return reconstructions, state, new_img


@torch.no_grad()
def visualize_rollout(
    env,
    actions,
    observations,
    world_model,
    logdir,
    max_path_length,
    use_env,
    forcing,
    tag,
    low_level_primitives,
    num_low_level_actions_per_primitive,
    primitive_model=None,
    policy=None,
    mode="eval",
    img_size=64,
    num_rollouts=4,
):
    file_path = logdir + "/"
    os.makedirs(file_path, exist_ok=True)
    print("Generating Rollout Visualization: ")

    if use_env:
        print("Generating Imagination Reconstructions forcing {}: ".format(forcing))
        file_suffix = "imagination_reconstructions.png"
    else:
        print("Generating Dataset Reconstructions {} forcing {}: ".format(tag, forcing))
        file_suffix = "dataset_reconstructions.png"
        file_suffix = tag + "_" + file_suffix
    if forcing != "none":
        file_suffix = forcing + "_forcing_" + file_suffix
    file_path += file_suffix

    with torch.cuda.amp.autocast():
        pl = max_path_length
        if low_level_primitives:
            pl = max_path_length * num_low_level_actions_per_primitive
        reconstructions = ptu.zeros(
            (
                num_rollouts,
                pl + 1,
                *world_model.image_shape,
            ),
        )
        obs = np.zeros(
            (
                num_rollouts,
                pl + 1,
                env.observation_space.shape[0],
            ),
            dtype=np.uint8,
        )
        for i in range(num_rollouts):
            state = world_model.initial(1)
            if low_level_primitives:
                # handling reset obs manually:
                if use_env:
                    a = ptu.zeros((1, env.low_level_action_dim))
                    o = env.reset()
                    policy.reset(o.reshape(1, -1))
                else:
                    a = actions[i, 0:1].to(ptu.device)
                    o = ptu.get_numpy(observations[i, 0])
                obs[i, 0] = o
                if primitive_model:
                    a = ptu.zeros((1, env.low_level_action_dim))
                state, _ = get_state(
                    o,
                    a,
                    state,
                    world_model,
                    forcing,
                    new_img=ptu.from_numpy(o),
                    first_output=True,
                )
                new_img = world_model.decode(world_model.get_features(state))
                reconstructions[i, 0] = new_img.unsqueeze(1)
                policy_o = (None, np.expand_dims(o, 0))
                for j in range(0, max_path_length):
                    if use_env:
                        high_level_action, _ = policy.get_action(policy_o)
                        o, r, d, info = env.step(
                            high_level_action[0],
                        )

                        ll_a = np.array(info["actions"])
                        ll_o = np.array(info["observations"])
                        ll_o = ll_o.transpose(0, 3, 1, 2).reshape(ll_o.shape[0], -1)

                        primitive_idx, primitive_args = (
                            np.argmax(high_level_action[0, : env.num_primitives]),
                            high_level_action[0, env.num_primitives :],
                        )
                        primitive_name = env.primitive_idx_to_name[primitive_idx]
                        ll_a, ll_o = subsample_paths(
                            ll_a,
                            ll_o,
                            ll_a.shape[0],
                            num_low_level_actions_per_primitive,
                        )
                        net = primitive_model
                        hl = high_level_action
                    else:
                        ll_a = ptu.get_numpy(
                            actions[
                                i,
                                1
                                + j * num_low_level_actions_per_primitive : 1
                                + (j + 1) * num_low_level_actions_per_primitive,
                            ]
                        )
                        ll_o = ptu.get_numpy(
                            observations[
                                i,
                                1
                                + j * num_low_level_actions_per_primitive : 1
                                + (j + 1) * num_low_level_actions_per_primitive,
                            ]
                        )
                        hl = None
                        net = None
                        primitive_name = None
                    policy_o = (np.expand_dims(ll_a, 0), np.expand_dims(ll_o, 0))
                    recons, state, new_img = forward_low_level_primitive(
                        ll_a,
                        ll_o,
                        world_model,
                        num_low_level_actions_per_primitive,
                        primitive_model,
                        net,
                        forcing,
                        new_img,
                        hl,
                        state,
                        primitive_name=primitive_name,
                    )
                    reconstructions[
                        i,
                        1
                        + j * num_low_level_actions_per_primitive : 1
                        + (j + 1) * num_low_level_actions_per_primitive,
                    ] = torch.cat(recons, dim=1)[0]
                    obs[
                        i,
                        1
                        + j * num_low_level_actions_per_primitive : 1
                        + (j + 1) * num_low_level_actions_per_primitive,
                    ] = np.array(ll_o)

            else:
                for j in range(0, max_path_length + 1):
                    if use_env:
                        if j == 0:
                            a = ptu.zeros((1, env.action_space.low.shape[0]))
                            o = env.reset()
                            new_img = ptu.from_numpy(o)
                            policy.reset(o.reshape(1, -1))
                        else:
                            high_level_action, _ = policy.get_action(o.reshape(1, -1))
                            o, r, d, info = env.step(
                                high_level_action[0],
                                render_every_step=True,
                                render_mode="rgb_array",
                                render_im_shape=(img_size, img_size),
                            )

                    else:
                        a = actions[i, j : j + 1].to(ptu.device)
                        o = ptu.get_numpy(observations[i, j])
                        new_img = ptu.from_numpy(o)

                    obs[i, j] = o
                    state, _ = get_state(
                        o, a, state, world_model, forcing, new_img, first_output=j == 0
                    )
                    new_img = world_model.decode(world_model.get_features(state))
                    reconstructions[i, j] = new_img.unsqueeze(1)

    reconstructions = torch.clamp(reconstructions + 0.5, 0, 1) * 255.0
    reconstructions = reconstructions.permute(0, 1, 3, num_rollouts, 2)
    reconstructions = ptu.get_numpy(reconstructions).astype(np.uint8)
    obs = ptu.from_numpy(obs)
    obs_np = ptu.get_numpy(
        obs.reshape(num_rollouts, pl + 1, 3, img_size, img_size).permute(
            0, 1, 3, num_rollouts, 2
        )
    ).astype(np.uint8)
    im = np.zeros((128 * num_rollouts, (pl + 1) * img_size, 3), dtype=np.uint8)

    for i in range(num_rollouts):
        for j in range(pl + 1):
            im[
                128 * i : 128 * i + img_size, img_size * j : img_size * (j + 1)
            ] = obs_np[i, j]
            im[
                128 * i + img_size : 128 * (i + 1), img_size * j : img_size * (j + 1)
            ] = reconstructions[i, j]
    cv2.imwrite(file_path, im)
    print("Saved Rollout Visualization to {}".format(file_path))

    file_path = osp.join(logdir, mode + "_video.avi")

    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(file_path, fourcc, 10.0, (img_size * 2, img_size * 2))
    for i in range(pl + 1):
        im1 = obs_np[0, i]
        im2 = obs_np[1, i]
        im3 = obs_np[2, i]
        imnum_rollouts = obs_np[3, i]

        im12 = np.concatenate((im1, im2), 1)
        im3num_rollouts = np.concatenate((im3, imnum_rollouts), 1)
        im = np.concatenate((im12, im3num_rollouts), 0)

        out.write(im)

    out.release()
    print("video saved to :", file_path)


def compute_world_model_loss(
    world_model,
    image_shape,
    image_dist,
    prior,
    post,
    prior_dist,
    post_dist,
    obs,
    forward_kl,
    free_nats,
    transition_loss_scale,
    kl_loss_scale,
    image_loss_scale,
):
    preprocessed_obs = world_model.flatten_obs(world_model.preprocess(obs), image_shape)
    image_pred_loss = -1 * image_dist.log_prob(preprocessed_obs).mean()
    post_detached_dist = world_model.get_detached_dist(post)
    prior_detached_dist = world_model.get_detached_dist(prior)
    if forward_kl:
        div = kld(post_dist, prior_dist).mean()
        div = torch.max(div, ptu.tensor(free_nats))
        prior_kld = kld(post_detached_dist, prior_dist).mean()
        post_kld = kld(post_dist, prior_detached_dist).mean()
    else:
        div = kld(prior_dist, post_dist).mean()
        div = torch.max(div, ptu.tensor(free_nats))
        prior_kld = kld(prior_dist, post_detached_dist).mean()
        post_kld = kld(prior_detached_dist, post_dist).mean()
    transition_loss = torch.max(prior_kld, ptu.tensor(free_nats))
    entropy_loss = torch.max(post_kld, ptu.tensor(free_nats))
    entropy_loss_scale = 1 - transition_loss_scale
    entropy_loss_scale = (1 - kl_loss_scale) * entropy_loss_scale
    transition_loss_scale = (1 - kl_loss_scale) * transition_loss_scale
    world_model_loss = (
        kl_loss_scale * div
        + image_loss_scale * image_pred_loss
        + transition_loss_scale * transition_loss
        + entropy_loss_scale * entropy_loss
    )
    return world_model_loss, div, image_pred_loss, transition_loss, entropy_loss


def world_model_loss_rt(
    world_model,
    image_shape,
    image_dist,
    reward_dist,
    prior,
    post,
    prior_dist,
    post_dist,
    pred_discount_dist,
    obs,
    rewards,
    terminals,
    forward_kl,
    free_nats,
    transition_loss_scale,
    kl_loss_scale,
    image_loss_scale,
    reward_loss_scale,
    pred_discount_loss_scale,
    discount,
):
    preprocessed_obs = world_model.flatten_obs(world_model.preprocess(obs), image_shape)
    image_pred_loss = -1 * image_dist.log_prob(preprocessed_obs).mean()
    post_detached_dist = world_model.get_detached_dist(post)
    prior_detached_dist = world_model.get_detached_dist(prior)
    reward_pred_loss = -1 * reward_dist.log_prob(rewards).mean()
    pred_discount_target = discount * (1 - terminals.float())
    pred_discount_loss = -1 * pred_discount_dist.log_prob(pred_discount_target).mean()
    if forward_kl:
        div = kld(post_dist, prior_dist).mean()
        div = torch.max(div, ptu.tensor(free_nats))
        prior_kld = kld(post_detached_dist, prior_dist).mean()
        post_kld = kld(post_dist, prior_detached_dist).mean()
    else:
        div = kld(prior_dist, post_dist).mean()
        div = torch.max(div, ptu.tensor(free_nats))
        prior_kld = kld(prior_dist, post_detached_dist).mean()
        post_kld = kld(prior_detached_dist, post_dist).mean()
    transition_loss = torch.max(prior_kld, ptu.tensor(free_nats))
    entropy_loss = torch.max(post_kld, ptu.tensor(free_nats))
    entropy_loss_scale = 1 - transition_loss_scale
    entropy_loss_scale = (1 - kl_loss_scale) * entropy_loss_scale
    transition_loss_scale = (1 - kl_loss_scale) * transition_loss_scale
    world_model_loss = (
        kl_loss_scale * div
        + image_loss_scale * image_pred_loss
        + transition_loss_scale * transition_loss
        + entropy_loss_scale * entropy_loss
        + reward_loss_scale * reward_pred_loss
        + pred_discount_loss_scale * pred_discount_loss
    )
    return (
        world_model_loss,
        div,
        image_pred_loss,
        reward_pred_loss,
        transition_loss,
        entropy_loss,
        pred_discount_loss,
    )


def update_network(network, optimizer, loss, gradient_clip, scaler):
    if type(network) == list:
        parameters = []
        for net in network:
            parameters.extend(list(net.parameters()))
    else:
        parameters = list(network.parameters())
    scaler.scale(loss).backward()
    if gradient_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, norm_type=2)
    scaler.step(optimizer)
    optimizer.zero_grad(set_to_none=True)


class NumpyDataset(Dataset):
    def __init__(
        self,
        inputs,
        outputs,
        batch_len,
        max_path_length,
        num_datapoints,
        randomize_batch_len=False,
    ):
        """
        :param inputs: (np.ndarray)
        :param outputs: (np.ndarray)
        :param max_path_length: (int)
        """
        self.inputs, self.outputs = inputs, outputs
        self.batch_len = batch_len
        self.max_path_length = max_path_length
        self.num_datapoints = num_datapoints
        self.randomize_batch_len = randomize_batch_len

    def __len__(self):
        """
        :return: (int)
        """
        return self.num_datapoints

    def __getitem__(self, i):
        """
        :param i: (int)
        :return (tuple, np.ndarray)
        """
        if type(self.inputs) == list or type(self.inputs) == tuple:
            inputs = []
            batch_start = np.random.randint(0, self.max_path_length - self.batch_len)
            idxs = np.linspace(
                batch_start,
                batch_start + self.batch_len,
                self.batch_len,
                endpoint=False,
            ).astype(int)
            for inp in self.inputs:
                if self.randomize_batch_len:
                    inputs.append(inp[i, idxs])
                else:
                    inputs.append(inp[i])
            if self.randomize_batch_len:
                outputs = self.outputs[i, idxs]
            else:
                outputs = self.outputs[i]
        else:
            batch_start = np.random.randint(0, self.max_path_length - self.batch_len)
            idxs = np.linspace(
                batch_start,
                batch_start + self.batch_len,
                self.batch_len,
                endpoint=False,
            ).astype(int)
            if self.randomize_batch_len:
                inputs = self.inputs[i, idxs]
            else:
                inputs = self.inputs[i]

            if self.randomize_batch_len:
                outputs = self.outputs[i, idxs]
            else:
                outputs = self.outputs[i]
        return inputs, outputs


def get_dataloader(
    filename,
    train_test_split,
    batch_len,
    batch_size,
    max_path_length,
    clone_primitives=False,
    randomize_batch_len=False,
):
    """
    :param filename: (str)
    :param train_test_split: (float)
    :param max_path_length: (int)
    :param clone_primitives_preprocess : (bool)
    :return: (tuple)
    """
    with h5py.File(filename, "r") as f:
        observations = np.array(f["observations"][:])
        actions = np.array(f["actions"][:])
        high_level_actions = np.array(f["high_level_actions"][:])
    argmax = np.argmax(high_level_actions[:, :, :10], axis=-1)
    one_hots = np.eye(10)[argmax]
    one_hots[:, 0:1, :] = np.zeros((one_hots.shape[0], 1, 10))
    high_level_actions = np.concatenate(
        (one_hots, high_level_actions[:, :, 10:]), axis=-1
    )
    num_train_datapoints = int(observations.shape[0] * train_test_split)
    observations, actions = torch.from_numpy(observations), torch.from_numpy(actions)
    high_level_actions = torch.from_numpy(high_level_actions)

    if clone_primitives:
        train_inputs, train_outputs = (
            high_level_actions[:num_train_datapoints],
            observations[:num_train_datapoints],
        ), actions[:num_train_datapoints]
        test_inputs, test_outputs = (
            high_level_actions[num_train_datapoints:],
            observations[num_train_datapoints:],
        ), actions[num_train_datapoints:]
    else:
        train_inputs, train_outputs = (
            actions[:num_train_datapoints],
            observations[:num_train_datapoints],
        )
        test_inputs, test_outputs = (
            actions[num_train_datapoints:],
            observations[num_train_datapoints:],
        )
    if randomize_batch_len:
        sampler_class = RandomSampler
    else:
        sampler_class = BatchLenRandomSampler

    train_dataset = NumpyDataset(
        train_inputs,
        train_outputs,
        batch_len,
        max_path_length,
        num_train_datapoints,
        randomize_batch_len=randomize_batch_len,
    )

    test_dataset = NumpyDataset(
        test_inputs,
        test_outputs,
        batch_len,
        max_path_length,
        observations.shape[0] - num_train_datapoints,
        randomize_batch_len=randomize_batch_len,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=sampler_class(train_dataset),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=sampler_class(test_dataset),
    )

    return train_dataloader, test_dataloader, train_dataset, test_dataset


def get_dataloader_rt(
    filename,
    train_test_split,
    batch_size,
    max_path_length,
):
    """
    :param filename: (str)
    :param train_test_split: (float)
    :param max_path_length: (int)
    :return: (tuple)
    """
    with h5py.File(filename, "r") as f:
        observations = np.array(f["observations"][:])
        actions = np.array(f["low_level_actions"][:])
        high_level_actions = np.array(f["high_level_actions"][:])
        rewards = np.array(f["rewards"][:])
        terminals = np.array(f["terminals"][:])
    argmax = np.argmax(high_level_actions[:, :, :10], axis=-1)
    one_hots = np.eye(10)[argmax]
    one_hots[:, 0:1, :] = np.zeros((one_hots.shape[0], 1, 10))
    high_level_actions = np.concatenate(
        (one_hots, high_level_actions[:, :, 10:]), axis=-1
    )
    num_train_datapoints = int(observations.shape[0] * train_test_split)
    observations, actions = torch.from_numpy(observations), torch.from_numpy(actions)
    high_level_actions = torch.from_numpy(high_level_actions)
    rewards, terminals = torch.from_numpy(rewards), torch.from_numpy(terminals)

    train_inputs, train_outputs = (
        high_level_actions[:num_train_datapoints],
        observations[:num_train_datapoints],
        rewards[:num_train_datapoints],
        terminals[:num_train_datapoints],
    ), actions[:num_train_datapoints]
    test_inputs, test_outputs = (
        high_level_actions[num_train_datapoints:],
        observations[num_train_datapoints:],
        rewards[num_train_datapoints:],
        terminals[num_train_datapoints:],
    ), actions[num_train_datapoints:]
    sampler_class = RandomSampler

    train_dataset = NumpyDataset(
        train_inputs,
        train_outputs,
        0,
        max_path_length,
        num_train_datapoints,
        randomize_batch_len=False,
    )

    test_dataset = NumpyDataset(
        test_inputs,
        test_outputs,
        0,
        max_path_length,
        observations.shape[0] - num_train_datapoints,
        randomize_batch_len=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=sampler_class(train_dataset),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=sampler_class(test_dataset),
    )

    return train_dataloader, test_dataloader, train_dataset, test_dataset


def get_dataloader_separately(
    filename,
    train_test_split,
    batch_len,
    batch_size,
    num_primitives,
    num_low_level_actions_per_primitive,
    env,
    randomize_batch_len=True,
):
    """
    :param filename: (str)
    :param train_test_split: (float)
    :param num_primitives: (int)
    :param num_low_level_actions_per_primitive: (int)
    :return: (tuple)
    """
    with h5py.File(filename, "r") as f:
        observations = np.array(f["observations"][:])
        actions = np.array(f["actions"][:])
        high_level_actions = np.array(f["high_level_actions"][:])

    obs = [[] for _ in range(num_primitives)]
    hls = [[] for _ in range(num_primitives)]
    acs = [[] for _ in range(num_primitives)]
    for i in range(observations.shape[0]):
        for j in range(1, observations.shape[1], num_low_level_actions_per_primitive):
            a = high_level_actions[i][j]
            primitive_idx, primitive_args = (
                np.argmax(a[: env.num_primitives]),
                high_level_actions[i][
                    j : j + num_low_level_actions_per_primitive, env.num_primitives :
                ],
            )
            primitive_name = env.primitive_idx_to_name[primitive_idx]
            primitive_actions = []
            for k in range(num_low_level_actions_per_primitive):
                primitive_name_to_action_dict = env.break_apart_action(
                    primitive_args[k]
                )
                primitive_action = primitive_name_to_action_dict[primitive_name]
                primitive_actions.append(primitive_action)
            primitive_actions = np.array(primitive_actions)
            if len(primitive_actions.shape) == 1:
                primitive_actions = primitive_actions.reshape(-1, 1)
            primitive_actions = np.concatenate(
                (
                    primitive_actions,
                    high_level_actions[i][
                        j : j + num_low_level_actions_per_primitive, -2:-1
                    ],
                ),
                axis=1,
            )
            obs[primitive_idx].append(
                observations[i][j : j + num_low_level_actions_per_primitive]
            )
            hls[primitive_idx].append(primitive_actions)
            acs[primitive_idx].append(
                actions[i][j : j + num_low_level_actions_per_primitive]
            )
    train_datasets = []
    test_datasets = []
    train_dataloaders = []
    test_dataloaders = []
    for p in range(num_primitives):
        ob = torch.from_numpy(
            np.concatenate([np.expand_dims(o, 0) for o in obs[p]], axis=0)
        )
        hl = torch.from_numpy(
            np.concatenate([np.expand_dims(h, 0) for h in hls[p]], axis=0)
        )
        ac = torch.from_numpy(
            np.concatenate([np.expand_dims(a, 0) for a in acs[p]], axis=0)
        )
        num_train_datapoints = int(ob.shape[0] * train_test_split)

        inputs, outputs = (hl[:num_train_datapoints], ob[:num_train_datapoints]), ac[
            :num_train_datapoints
        ]
        if randomize_batch_len:
            sampler_class = RandomSampler
        else:
            sampler_class = BatchLenRandomSampler

        train_dataset = NumpyDataset(
            inputs,
            outputs,
            batch_len,
            num_low_level_actions_per_primitive + 1,
            num_train_datapoints,
            randomize_batch_len=randomize_batch_len,
        )

        inputs, outputs = (hl[num_train_datapoints:], ob[num_train_datapoints:]), ac[
            num_train_datapoints:
        ]

        test_dataset = NumpyDataset(
            inputs,
            outputs,
            batch_len,
            num_low_level_actions_per_primitive + 1,
            ob.shape[0] - num_train_datapoints,
            randomize_batch_len=randomize_batch_len,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler_class(train_dataset),
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler_class(test_dataset),
        )

        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
        train_dataloaders.append(train_dataloader)
        test_dataloaders.append(test_dataloader)

        print(p, hl.shape)
    return train_dataloaders, test_dataloaders, train_datasets, test_datasets
