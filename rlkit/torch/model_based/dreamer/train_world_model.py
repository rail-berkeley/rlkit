import os

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
):
    file_path = logdir + "/plots/"
    os.makedirs(file_path, exist_ok=True)

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
        if low_level_primitives:
            reconstructions = ptu.zeros(
                (
                    4,
                    max_path_length * num_low_level_actions_per_primitive + 1,
                    *world_model.image_shape,
                ),
            )
            obs = np.zeros(
                (
                    4,
                    max_path_length * num_low_level_actions_per_primitive + 1,
                    env.observation_space.shape[0],
                ),
                dtype=np.uint8,
            )
        else:
            reconstructions = ptu.zeros(
                (4, max_path_length + 1, *world_model.image_shape),
            )
            obs = np.zeros(
                (4, max_path_length + 1, env.observation_space.shape[0]),
                dtype=np.uint8,
            )
        for i in range(4):
            state = world_model.initial(1)
            if low_level_primitives:
                # handling reset obs manually:
                if use_env:
                    a = ptu.zeros((1, 9))
                    o = env.reset()
                else:
                    a = actions[i, 0:1].to(ptu.device)
                    o = ptu.get_numpy(observations[i, 0])
                obs[i, 0] = o

                if primitive_model:
                    a = ptu.zeros((1, 9))
                if forcing == "teacher":
                    o = ptu.from_numpy(o.reshape(1, -1))
                    embed = world_model.encode(o)
                    state, prior = world_model.obs_step(state, a, embed)
                elif forcing == "self" and world_model.use_prior_instead_of_posterior:
                    state = world_model.action_step(state, a)
                    prior = None
                else:
                    state = world_model.action_step(state, a)
                    prior = None
                    if prior is not None and world_model.use_prior_instead_of_posterior:
                        state = prior
                new_img = world_model.decode(world_model.get_features(state))
                reconstructions[i, 0] = new_img.unsqueeze(1)
                for j in range(0, max_path_length):
                    if use_env:
                        high_level_action = np.array([env.action_space.sample()])
                        o, r, d, info = env.step(
                            high_level_action[0],
                        )

                        ll_a = np.array(info["actions"])
                        ll_o = np.array(info["observations"])
                        ll_o = ll_o.transpose(0, 3, 1, 2).reshape(ll_o.shape[0], -1)

                        num_ll = ll_a.shape[0]
                        idxs = np.linspace(
                            0, num_ll, num_low_level_actions_per_primitive + 1
                        )
                        spacing = num_ll // (num_low_level_actions_per_primitive)
                        a = ll_a.reshape(
                            num_low_level_actions_per_primitive, spacing, -1
                        )
                        a = a.sum(axis=1)[:, :3]  # just keep sum of xyz deltas
                        ll_a = np.concatenate(
                            (a, ll_a[idxs.astype(np.int)[0:-1], 3:]), axis=1
                        )
                        ll_a = ptu.from_numpy(ll_a)
                        ll_o = ll_o[idxs.astype(np.int)[1:] - 1]
                    else:
                        ll_a = actions[
                            i,
                            1
                            + j * num_low_level_actions_per_primitive : 1
                            + (j + 1) * num_low_level_actions_per_primitive,
                        ].to(ptu.device)
                        ll_o = ptu.get_numpy(
                            observations[
                                i,
                                1
                                + j * num_low_level_actions_per_primitive : 1
                                + (j + 1) * num_low_level_actions_per_primitive,
                            ]
                        )
                    for k in range(0, num_low_level_actions_per_primitive):
                        a = ll_a[k : k + 1]
                        o = ll_o[k]
                        obs[i, 1 + j * num_low_level_actions_per_primitive + k] = o
                        if primitive_model:
                            tmp = np.array(
                                [
                                    (j * k + 1)
                                    / (
                                        num_low_level_actions_per_primitive
                                        * max_path_length
                                    )
                                ]
                            ).reshape(1, -1)
                            hl = np.concatenate((high_level_action, tmp), 1)
                            state = world_model(
                                ptu.from_numpy(o.reshape(1, 1, o.shape[-1])),
                                (ptu.from_numpy(hl.reshape(1, 1, hl.shape[-1])), None),
                                primitive_model,
                                use_network_action=True,
                            )[0]
                        else:
                            if forcing == "teacher":
                                o = ptu.from_numpy(o.reshape(1, -1))
                                embed = world_model.encode(o)
                                state, prior = world_model.obs_step(state, a, embed)
                            elif (
                                forcing == "self"
                                and world_model.use_prior_instead_of_posterior
                            ):
                                o = new_img.reshape(-1, obs.shape[-1])
                                o = torch.clamp(o + 0.5, 0, 1) * 255.0
                                embed = world_model.encode(o)
                                state, prior = world_model.obs_step(state, a, embed)
                            else:
                                state = world_model.action_step(state, a)
                                prior = None
                            if (
                                prior is not None
                                and world_model.use_prior_instead_of_posterior
                            ):
                                state = prior
                        new_img = world_model.decode(world_model.get_features(state))
                        reconstructions[
                            i, 1 + j * num_low_level_actions_per_primitive + k
                        ] = new_img.unsqueeze(1)
            else:
                for j in range(0, max_path_length + 1):
                    if use_env:
                        if j == 0:
                            a = ptu.zeros((1, env.action_space.low.shape[0]))
                            o = env.reset()
                        else:
                            a = ptu.from_numpy(np.array([env.action_space.sample()]))
                            o, r, d, info = env.step(
                                a[0].detach().cpu().numpy(),
                            )
                    else:
                        a = actions[i, j : j + 1].to(ptu.device)
                        o = ptu.get_numpy(observations[i, j])
                    obs[i, j] = o
                    if forcing == "teacher":
                        o = ptu.from_numpy(o.reshape(1, -1))
                        embed = world_model.encode(o)
                        state, prior = world_model.obs_step(state, a, embed)
                    elif (
                        forcing == "self" and world_model.use_prior_instead_of_posterior
                    ):
                        if j == 0:
                            state = world_model.action_step(state, a)
                            prior = None
                        else:
                            o = new_img.reshape(-1, obs.shape[-1])
                            o = torch.clamp(o + 0.5, 0, 1) * 255.0
                            embed = world_model.encode(o)
                            state, prior = world_model.obs_step(state, a, embed)
                    else:
                        state = world_model.action_step(state, a)
                        prior = None
                    if prior is not None and world_model.use_prior_instead_of_posterior:
                        state = prior
                    new_img = world_model.decode(world_model.get_features(state))
                    reconstructions[i, j] = new_img.unsqueeze(1)

    reconstructions = (
        torch.clamp(reconstructions.permute(0, 1, 3, 4, 2) + 0.5, 0, 1) * 255.0
    )
    reconstructions = ptu.get_numpy(reconstructions).astype(np.uint8)
    obs = ptu.from_numpy(obs)
    if low_level_primitives:
        obs_np = ptu.get_numpy(
            obs.reshape(
                4, max_path_length * num_low_level_actions_per_primitive + 1, 3, 64, 64
            ).permute(0, 1, 3, 4, 2)
        ).astype(np.uint8)
        im = np.zeros(
            (
                128 * 4,
                (max_path_length * num_low_level_actions_per_primitive + 1) * 64,
                3,
            ),
            dtype=np.uint8,
        )
        for i in range(4):
            for j in range(max_path_length * num_low_level_actions_per_primitive + 1):
                im[128 * i : 128 * i + 64, 64 * j : 64 * (j + 1)] = obs_np[i, j]
                im[
                    128 * i + 64 : 128 * (i + 1), 64 * j : 64 * (j + 1)
                ] = reconstructions[i, j]
    else:
        obs_np = ptu.get_numpy(
            obs.reshape(4, max_path_length + 1, 3, 64, 64).permute(0, 1, 3, 4, 2)
        ).astype(np.uint8)
        im = np.zeros((128 * 4, (max_path_length + 1) * 64, 3), dtype=np.uint8)

        for i in range(4):
            for j in range(max_path_length + 1):
                im[128 * i : 128 * i + 64, 64 * j : 64 * (j + 1)] = obs_np[i, j]
                im[
                    128 * i + 64 : 128 * (i + 1), 64 * j : 64 * (j + 1)
                ] = reconstructions[i, j]
    cv2.imwrite(file_path, im)


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
        self.num_datapoints
        return self.num_datapoints

    def __getitem__(self, i):
        """
        :param i: (int)
        :return (tuple, np.ndarray)
        """
        if type(self.inputs) != np.ndarray:
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
            inputs, outputs = self.inputs[i], self.outputs[i]
        return inputs, outputs


def clone_primitives_preprocess_fn(observations, actions):
    """
    :param observations: (np.ndarray)
    :param actions: (np.ndarray)
    :return: (tuple)
    """
    observations = observations[:, :, :]
    actions = actions[:, :, :]
    return observations, actions


def get_dataloader(
    filename,
    train_test_split,
    batch_len,
    batch_size,
    max_path_length,
    clone_primitives_preprocess=False,
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
        if clone_primitives_preprocess:
            high_level_actions = torch.from_numpy(np.array(f["high_level_actions"][:]))
    observations, actions = torch.from_numpy(observations), torch.from_numpy(actions)
    if clone_primitives_preprocess:
        observations, actions = clone_primitives_preprocess_fn(observations, actions)
        inputs, outputs = (high_level_actions, observations), actions
        randomize_batch_len = True
        sampler_class = RandomSampler
    else:
        inputs, outputs = actions, observations
        randomize_batch_len = False
        sampler_class = BatchLenRandomSampler
    num_train_datapoints = int(observations.shape[0] * train_test_split)

    train_dataset = NumpyDataset(
        inputs,
        outputs,
        batch_len,
        max_path_length,
        num_train_datapoints,
        randomize_batch_len=randomize_batch_len,
    )
    test_dataset = NumpyDataset(
        inputs,
        outputs,
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

    print(actions.shape)
    print(
        actions.mean().item(),
        actions.std().item(),
        actions.min().item(),
        actions.max().item(),
    )

    return train_dataloader, test_dataloader, train_dataset, test_dataset
