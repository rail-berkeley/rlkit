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


def reconstruct_from_state(state, world_model):
    feat = world_model.get_features(state)
    feat = feat.reshape(-1, feat.shape[-1])
    new_img = (torch.clamp(world_model.decode(feat) + 0.5, 0, 1) * 255.0).type(
        torch.ByteTensor
    )
    new_img = ptu.get_numpy(new_img.permute(0, 2, 3, 1)[0]).astype(np.uint8)
    new_img = np.ascontiguousarray(np.copy(new_img), dtype=np.uint8)
    return new_img


def convert_img_to_save(img):
    img = np.copy(img.reshape(3, 64, 64).transpose(1, 2, 0))
    img = np.ascontiguousarray(img, dtype=np.uint8)
    return img


def add_text(vis, text, pos, scale, rgb):
    cv2.putText(
        vis,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        rgb,
        1,
        cv2.LINE_AA,
    )


@torch.no_grad()
def visualize_rollout(
    env,
    world_model,
    logdir,
    max_path_length,
    low_level_primitives,
    policy=None,
    mode="eval",
    img_size=64,
    num_rollouts=4,
    use_raps_obs=False,
    use_true_actions=True,
):
    file_path = logdir + "/"
    os.makedirs(file_path, exist_ok=True)
    print(
        "Generating Imagination Reconstructions No Intermediate Obs: {} Actual Actions: {}".format(
            use_raps_obs, use_true_actions
        )
    )
    file_suffix = "imagination_reconstructions_raps_obs_{}_true_actions.png".format(
        use_raps_obs, use_true_actions
    )
    file_path += file_suffix
    pl = max_path_length
    img_shape = (img_size, img_size, 3)
    reconstructions = np.zeros(
        (num_rollouts, pl + 1, *img_shape),
        dtype=np.uint8,
    )
    obs = np.zeros(
        (num_rollouts, pl + 1, *img_shape),
        dtype=np.uint8,
    )
    with torch.cuda.amp.autocast():
        for i in range(num_rollouts):
            for j in range(0, max_path_length + 1):
                if j == 0:
                    o = env.reset()
                    new_img = ptu.from_numpy(o)
                    policy.reset(o.reshape(1, -1))
                    r = 0
                    if low_level_primitives:
                        policy_o = (None, o.reshape(1, -1))
                    else:
                        policy_o = o.reshape(1, -1)
                    vis = convert_img_to_save(o)
                    add_text(vis, "Ground Truth", (1, 60), 0.25, (0, 255, 0))
                else:
                    high_level_action, state = policy.get_action(
                        policy_o, use_raps_obs, use_true_actions
                    )
                    state = state["state"]
                    o, r, d, info = env.step(
                        high_level_action[0],
                    )
                    if low_level_primitives:
                        ll_o = np.expand_dims(np.array(info["observations"]), 0)
                        ll_a = np.expand_dims(np.array(info["actions"]), 0)
                        policy_o = (ll_a, ll_o)
                    else:
                        policy_o = o.reshape(1, -1)
                    (
                        primitive_name,
                        _,
                        _,
                    ) = env.get_primitive_info_from_high_level_action(
                        high_level_action[0]
                    )
                    vis = convert_img_to_save(o)
                    add_text(vis, primitive_name, (1, 60), 0.25, (0, 255, 0))
                    add_text(vis, "r: {}".format(r), (35, 7), 0.3, (0, 0, 0))

                obs[i, j] = vis
                if j != 0:
                    new_img = reconstruct_from_state(state, world_model)
                    if j == 1:
                        add_text(new_img, "Reconstruction", (1, 60), 0.25, (0, 255, 0))
                    reconstructions[i, j - 1] = new_img
                    print(
                        "Rollout {} Step {} Predicted Reward".format(i, j - 1),
                        world_model.reward(world_model.get_features(state))
                        .detach()
                        .cpu()
                        .numpy()
                        .item(),
                    )
                    print("Rollout {} Step {} Reward".format(i, j - 1), prev_r)
                    print(
                        "Rollout {} Step {} Predicted Discount".format(i, j - 1),
                        world_model.get_dist(
                            world_model.pred_discount(world_model.get_features(state)),
                            std=None,
                            normal=False,
                        )
                        .mean.detach()
                        .cpu()
                        .numpy()
                        .item(),
                    )
                    print()
                prev_r = r
            _, state = policy.get_action(policy_o)
            state = state["state"]
            new_img = reconstruct_from_state(state, world_model)
            reconstructions[i, max_path_length] = new_img
            print(
                "Rollout {} Final Predicted Reward".format(i),
                world_model.reward(world_model.get_features(state))
                .detach()
                .cpu()
                .numpy()
                .item(),
            )
            print("Rollout {} Final Reward: ".format(i), r)
            print(
                "Rollout {} Final Predicted Discount".format(i, j - 1),
                world_model.get_dist(
                    world_model.pred_discount(world_model.get_features(state)),
                    std=None,
                    normal=False,
                )
                .mean.detach()
                .cpu()
                .numpy()
                .item(),
            )
            print()

    im = np.zeros((img_size * 2 * num_rollouts, (pl + 1) * img_size, 3), dtype=np.uint8)

    for i in range(num_rollouts):
        for j in range(pl + 1):
            im[
                img_size * 2 * i : img_size * 2 * i + img_size,
                img_size * j : img_size * (j + 1),
            ] = obs[i, j]
            im[
                img_size * 2 * i + img_size : img_size * 2 * (i + 1),
                img_size * j : img_size * (j + 1),
            ] = reconstructions[i, j]
    cv2.imwrite(file_path, im)
    print("Saved Rollout Visualization to {}".format(file_path))

    # file_path = osp.join(logdir, mode + "_video.avi")

    # fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    # out = cv2.VideoWriter(file_path, fourcc, 10.0, (img_size * 2, img_size * 2))
    # for i in range(pl + 1):
    #     im1 = obs[0, i]
    #     im2 = obs[1, i]
    #     im3 = obs[2, i]
    #     imnum_rollouts = obs[3, i]

    #     im12 = np.concatenate((im1, im2), 1)
    #     im3num_rollouts = np.concatenate((im3, imnum_rollouts), 1)
    #     im = np.concatenate((im12, im3num_rollouts), 0)

    #     out.write(im)

    # out.release()
    # print("video saved to :", file_path)
    print()


def unsubsample_and_execute_ll(ll_a, env, num_subsample_steps):
    for z in range(ll_a.shape[0]):
        la = ll_a[z]
        target = env.get_endeff_pos() + la[:3]
        for k in range(num_subsample_steps):
            a = (target - env.get_endeff_pos()) / num_subsample_steps
            a = np.concatenate((a, la[3:]))
            o, r, d, i = env.low_level_step(a)
    return o, r, d, i


@torch.no_grad()
def visualize_primitive_unsubsampled_rollout(
    env1,
    env2,
    env3,
    logdir,
    max_path_length,
    num_low_level_actions_per_primitive,
    policy=None,
    img_size=64,
    num_rollouts=4,
):
    file_path = logdir + "/"
    os.makedirs(file_path, exist_ok=True)
    print("Generating Unsubsampled Rollout")
    file_suffix = "unsubsampled_rollout.png"
    file_path += file_suffix
    pl = max_path_length
    img_shape = (img_size, img_size, 3)
    obs1 = np.zeros(
        (num_rollouts, pl + 1, *img_shape),
        dtype=np.uint8,
    )
    obs2 = np.zeros(
        (num_rollouts, pl + 1, *img_shape),
        dtype=np.uint8,
    )
    obs3 = np.zeros(
        (num_rollouts, pl + 1, *img_shape),
        dtype=np.uint8,
    )
    with torch.cuda.amp.autocast():
        for i in range(num_rollouts):
            for j in range(0, max_path_length + 1):
                if j == 0:
                    o1 = env1.reset()
                    o2 = env2.reset()
                    o3 = env3.reset()
                    policy_o = (None, o1.reshape(1, -1))
                    policy.reset(policy_o[1])
                    o1 = convert_img_to_save(o1)
                    o2 = convert_img_to_save(o2)
                    o3 = convert_img_to_save(o3)
                    add_text(o1, "True", (1, 60), 0.25, (0, 255, 0))
                    add_text(o2, "True Unsubsampled", (1, 60), 0.25, (0, 255, 0))
                    add_text(o3, "Pred Unsubsampled", (1, 60), 0.25, (0, 255, 0))
                    obs1[i, 0] = o1
                    obs2[i, 0] = o2
                    obs3[i, 0] = o3

                else:
                    high_level_action, out = policy.get_action(policy_o)
                    ll_a_pred = out["ll_a_pred"]
                    o1, _, _, i1 = env1.step(
                        high_level_action[0],
                    )
                    ll_o = np.expand_dims(np.array(i1["observations"]), 0)
                    ll_a = np.expand_dims(np.array(i1["actions"]), 0)
                    policy_o = (ll_a, ll_o)
                    (
                        primitive_name,
                        _,
                        primitive_idx,
                    ) = env1.get_primitive_info_from_high_level_action(
                        high_level_action[0]
                    )
                    num_subsample_steps = (
                        env1.primitive_idx_to_num_low_level_steps[primitive_idx]
                        // num_low_level_actions_per_primitive
                    )
                    o2, _, _, i2 = unsubsample_and_execute_ll(
                        ll_a[0], env2, num_subsample_steps
                    )
                    if j > 1:
                        o3, _, _, i3 = unsubsample_and_execute_ll(
                            ll_a_pred, env3, num_subsample_steps
                        )
                        obs3[i, j - 1] = convert_img_to_save(o3)
                        print(
                            "Rollout: {}".format(i),
                            "Step: {}".format(j - 1),
                            "Primitive: ",
                            prev_primitive_name,
                            "LL Action Pred Error: ",
                            (np.linalg.norm(prev_ll_a - ll_a_pred) ** 2)
                            / num_low_level_actions_per_primitive,
                        )
                    prev_ll_a = ll_a
                    prev_primitive_name = primitive_name
                    obs1[i, j] = convert_img_to_save(o1)
                    obs2[i, j] = convert_img_to_save(o2)
            _, out = policy.get_action(policy_o)
            ll_a_pred = out["ll_a_pred"]
            o3, _, _, i3 = unsubsample_and_execute_ll(
                ll_a_pred, env3, num_subsample_steps
            )
            obs3[i, max_path_length] = convert_img_to_save(o3)
            print(
                "Rollout: {}".format(i),
                "Step: {}".format(max_path_length),
                "Primitive: ",
                prev_primitive_name,
                "LL Action Pred Error: ",
                (np.linalg.norm(prev_ll_a - ll_a_pred) ** 2)
                / num_low_level_actions_per_primitive,
            )
            print("Rollout {} Final Success True Actions: ".format(i), i1["success"])
            print(
                "Rollout {} Final Success True Actions Unsubsampled: ".format(i),
                i2["success"],
            )
            print(
                "Rollout {} Final Success Primitive Model Actions Unsubsampled: ".format(
                    i
                ),
                i3["success"],
            )
            print()

    im = np.zeros((img_size * 3 * num_rollouts, (pl + 1) * img_size, 3), dtype=np.uint8)

    for i in range(num_rollouts):
        for j in range(pl + 1):
            im[
                img_size * 3 * i : img_size * 3 * i + img_size,
                img_size * j : img_size * (j + 1),
            ] = obs1[i, j]
            im[
                img_size * 3 * i + img_size : img_size * 3 * i + 2 * img_size,
                img_size * j : img_size * (j + 1),
            ] = obs2[i, j]
            im[
                img_size * 3 * i + 2 * img_size : img_size * 3 * i + 3 * img_size,
                img_size * j : img_size * (j + 1),
            ] = obs3[i, j]
    cv2.imwrite(file_path, im)
    print("Saved Rollout Visualization to {}".format(file_path))


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
