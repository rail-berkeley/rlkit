import os

import cv2
import h5py
import matplotlib
import numpy as np
import torch
from torch.distributions import kl_divergence as kld
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

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
    env, dataset, world_model, logdir, max_path_length, epoch, use_env, forcing, tag
):
    file_path = logdir + "/plots/"
    os.makedirs(file_path, exist_ok=True)

    if use_env:
        print("Generating Imagination Reconstructions forcing {}: ".format(forcing))
        file_suffix = "imagination_reconstructions_{}.png".format(epoch)
    else:
        print("Generating Dataset Reconstructions {} forcing {}: ".format(tag, forcing))
        file_suffix = "dataset_reconstructions_{}.png".format(epoch)
        file_suffix = tag + "_" + file_suffix
    if forcing != "none":
        file_suffix = forcing + "_forcing_" + file_suffix
    file_path += file_suffix

    with torch.cuda.amp.autocast():
        reconstructions = ptu.zeros(
            (4, max_path_length + 1, *world_model.image_shape),
        )
        obs = np.zeros(
            (4, max_path_length + 1, env.observation_space.shape[0]),
            dtype=np.uint8,
        )
        for i in range(4):
            state = world_model.initial(1)
            for j in range(0, max_path_length + 1):
                if use_env:
                    if j == 0:
                        a = ptu.zeros((1, env.action_space.low.shape[0]))
                    else:
                        a = ptu.from_numpy(np.array([env.action_space.sample()]))
                else:
                    a = dataset.actions[i, j : j + 1].to(ptu.device)
                if use_env:
                    if j == 0:
                        o = env.reset()
                    else:
                        o, r, d, _ = env.step(
                            a[0].detach().cpu().numpy(),
                        )
                    obs[i, j] = o
                else:
                    o = ptu.get_numpy(dataset.observations[i, j])
                    obs[i, j] = o
                if forcing == "teacher":
                    o = ptu.from_numpy(o.reshape(1, -1))
                    embed = world_model.encode(o)
                    state, prior = world_model.obs_step(state, a, embed)
                elif forcing == "self" and world_model.use_prior_instead_of_posterior:
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
    obs_np = ptu.get_numpy(
        obs.reshape(4, max_path_length + 1, 3, 64, 64).permute(0, 1, 3, 4, 2)
    ).astype(np.uint8)

    im = np.zeros((128 * 4, (max_path_length + 1) * 64, 3), dtype=np.uint8)
    for i in range(4):
        for j in range(max_path_length + 1):
            im[128 * i : 128 * i + 64, 64 * j : 64 * (j + 1)] = obs_np[i, j]
            im[128 * i + 64 : 128 * (i + 1), 64 * j : 64 * (j + 1)] = reconstructions[
                i, j
            ]
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
    def __init__(self, observations, actions, batch_len, max_path_length):
        """
        :param observations: (np.ndarray)
        :param actions: (np.ndarray)
        :param max_path_length: (int)
        """
        self.observations, self.actions = observations, actions
        self.batch_len = batch_len
        self.max_path_length = max_path_length + 1

    def __len__(self):
        """
        :return: (int)
        """
        return self.observations.shape[0]

    def __getitem__(self, i):
        """
        :param i: (int)
        :return (tuple, np.ndarray)
        """
        obs, actions = self.observations[i], self.actions[i]
        return obs, actions


def get_dataloader(filename, train_test_split, batch_len, batch_size, max_path_length):
    """
    :param filename: (str)
    :param train_test_split: (float)
    :param max_path_length: (int)
    :return: (tuple)
    """
    with h5py.File(filename, "r") as f:
        observations = np.array(f["observations"][:])
        actions = np.array(f["actions"][:])
    observations, actions = torch.from_numpy(observations), torch.from_numpy(actions)
    num_train_datapoints = int(observations.shape[0] * train_test_split)
    train_dataset = NumpyDataset(
        observations[:num_train_datapoints],
        actions[:num_train_datapoints],
        batch_len,
        max_path_length,
    )
    test_dataset = NumpyDataset(
        observations[num_train_datapoints:],
        actions[num_train_datapoints:],
        batch_len,
        max_path_length,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=BatchLenRandomSampler(train_dataset),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=BatchLenRandomSampler(test_dataset),
    )
    return train_dataloader, test_dataloader, train_dataset, test_dataset
