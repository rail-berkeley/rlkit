import os

import cv2
import h5py
import matplotlib
import numpy as np
import torch
import torch.optim as optim
from arguments import get_args
from torch.distributions import kl_divergence as kld
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.primitives_make_env import make_env
from rlkit.torch.model_based.dreamer.world_models import WorldModel

matplotlib.use("Agg")
from matplotlib import pyplot as plt


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
def imagination_post_epoch_func(env, world_model, args, max_path_length, epoch):
    print("Generating Imagination Reconstructions: ")
    with torch.cuda.amp.autocast():
        null_state = world_model.initial(4)
        null_acts = ptu.zeros((4, env.action_space.low.shape[0]))
        reset_obs = []
        for i in range(4):
            reset_obs.append(env.reset().reshape(1, -1))
        reset_obs = ptu.from_numpy(np.concatenate(reset_obs))
        embed = world_model.encode(reset_obs)
        new_state, _ = world_model.obs_step(null_state, null_acts, embed)
        reconstructions = ptu.zeros(
            (4, max_path_length, *world_model.image_shape),
        )
        actions = ptu.zeros((4, max_path_length, env.action_space.low.size))
        for k in range(max_path_length):
            action = ptu.from_numpy(
                np.array([env.action_space.sample() for i in range(4)])
            )
            new_state = world_model.action_step(new_state, action)
            new_img = world_model.decode(world_model.get_features(new_state))
            reconstructions[:, k : k + 1] = new_img.unsqueeze(1)
            actions[:, k : k + 1] = action.unsqueeze(1)
        obs = np.zeros(
            (4, max_path_length, env.observation_space.shape[0]),
            dtype=np.uint8,
        )
        for i in range(4):
            env.reset()
            o = env.reset()
            for j in range(max_path_length):
                o, r, d, _ = env.step(
                    actions[i, j].detach().cpu().numpy(),
                )
                obs[i, j] = o
    reconstructions = (
        torch.clamp(
            reconstructions.permute(0, 1, 3, 4, 2) + 0.5,
            0,
            1,
        )
        * 255.0
    )
    reconstructions = ptu.get_numpy(reconstructions).astype(np.uint8)
    obs = ptu.from_numpy(obs)
    obs_np = ptu.get_numpy(
        obs[:, :, : 64 * 64 * 3]
        .reshape(4, max_path_length, 3, 64, 64)
        .permute(0, 1, 3, 4, 2)
    ).astype(np.uint8)
    file_path = (
        "data/"
        + args.logdir
        + "/plots/imagination_reconstructions_{}.png".format(epoch)
    )
    im = np.zeros((128 * 4, max_path_length * 64, 3), dtype=np.uint8)
    for i in range(4):
        for j in range(max_path_length):
            im[128 * i : 128 * i + 64, 64 * j : 64 * (j + 1)] = obs_np[i, j]
            im[128 * i + 64 : 128 * (i + 1), 64 * j : 64 * (j + 1)] = reconstructions[
                i, j
            ]
    cv2.imwrite(file_path, im)


def visualize_dataset_trajectory(
    env, dataset, world_model, args, max_path_length, epoch, tag
):
    print("Generating {} Reconstructions: ", tag)
    with torch.cuda.amp.autocast():
        obs = dataset.observations[:4].to(ptu.device)
        actions = dataset.actions[:4].to(ptu.device)
        null_state = world_model.initial(4)
        acts = actions[:, 0]
        init_obs = obs[:, 0]
        embed = world_model.encode(init_obs)
        new_state, _ = world_model.obs_step(null_state, acts, embed)
        new_img = world_model.decode(world_model.get_features(new_state))
        reconstructions = ptu.zeros(
            (4, max_path_length + 1, *world_model.image_shape),
        )
        reconstructions[:, 0:1] = new_img.unsqueeze(1)
        for k in range(1, max_path_length + 1):
            action = actions[:, k]
            new_state = world_model.action_step(new_state, action)
            new_img = world_model.decode(world_model.get_features(new_state))
            reconstructions[:, k : k + 1] = new_img.unsqueeze(1)
    reconstructions = (
        torch.clamp(
            reconstructions.permute(0, 1, 3, 4, 2) + 0.5,
            0,
            1,
        )
        * 255.0
    )
    reconstructions = ptu.get_numpy(reconstructions).astype(np.uint8)
    obs_np = ptu.get_numpy(
        obs.reshape(4, max_path_length + 1, 3, 64, 64).permute(0, 1, 3, 4, 2)
    ).astype(np.uint8)
    file_path = (
        "data/" + args.logdir + "/plots/{}_reconstructions_{}.png".format(tag, epoch)
    )
    im = np.zeros((128 * 4, max_path_length * 64, 3), dtype=np.uint8)
    for i in range(4):
        for j in range(max_path_length):
            im[128 * i : 128 * i + 64, 64 * j : 64 * (j + 1)] = obs_np[i, j]
            im[128 * i + 64 : 128 * (i + 1), 64 * j : 64 * (j + 1)] = reconstructions[
                i, j
            ]
    cv2.imwrite(file_path, im)
    if tag == "test":
        import ipdb

        ipdb.set_trace()


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

    return (
        world_model_loss,
        div,
        image_pred_loss,
        transition_loss,
        entropy_loss,
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


def preprocess_data(observations, actions, max_path_length):
    o = []
    a = []
    for l in observations:
        for i in range(max_path_length):
            o.extend(l[i])

    for l in actions:
        for i in range(max_path_length):
            a.append(np.zeros(9))
            a.extend(l[i])

    o = np.array(o).transpose(0, 3, 1, 2)
    a = np.array(a)
    print(o.shape, a.shape)
    return o, a


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


def get_dataloader(filename, args):
    """
    :param filename: (str)
    :param train_test_split: (float)
    :param max_path_length: (int)
    :return: (tuple)
    """
    args = get_args()
    with h5py.File(filename, "r") as f:
        observations = np.array(f["observations"][:])
        actions = np.array(f["actions"][:])
    observations, actions = torch.from_numpy(observations), torch.from_numpy(actions)
    import ipdb

    ipdb.set_trace()
    num_train_datapoints = int(observations.shape[0] * args.train_test_split)
    train_dataset = NumpyDataset(
        observations[:num_train_datapoints],
        actions[:num_train_datapoints],
        args.batch_len,
        args.max_path_length,
    )
    test_dataset = NumpyDataset(
        observations[num_train_datapoints:],
        actions[num_train_datapoints:],
        args.batch_len,
        args.max_path_length,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        sampler=BatchLenRandomSampler(train_dataset),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        sampler=BatchLenRandomSampler(test_dataset),
    )
    return train_dataloader, test_dataloader, train_dataset, test_dataset


if __name__ == "__main__":
    args = get_args()
    ptu.device = torch.device("cuda")
    scaler = torch.cuda.amp.GradScaler()

    env_kwargs = dict(
        control_mode=args.control_mode,
        action_scale=1,
        max_path_length=args.max_path_length,
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
            max_path_length=args.max_path_length,
            unflatten_images=False,
        ),
        image_kwargs=dict(imwidth=64, imheight=64),
    )

    env_suite = "metaworld"
    env_name = "reach-v2"
    env = make_env(env_suite, env_name, env_kwargs)

    world_model_loss_kwargs = dict(
        forward_kl=False,
        free_nats=1.0,
        transition_loss_scale=0.8,
        kl_loss_scale=0.0,
        image_loss_scale=1.0,
    )
    world_model_kwargs = dict(
        model_hidden_size=400,
        stochastic_state_size=50,
        deterministic_state_size=200,
        embedding_size=1024,
        rssm_hidden_size=200,
        reward_num_layers=2,
        pred_discount_num_layers=3,
        gru_layer_norm=True,
        std_act="sigmoid2",
        action_dim=env.action_space.low.shape[0],
        image_shape=env.image_shape,
    )

    optimizer_kwargs = dict(
        lr=3e-4,
        eps=1e-5,
        weight_decay=0.0,
    )
    gradient_clip = 100

    world_model = WorldModel(
        **world_model_kwargs,
    ).to(ptu.device)

    image_shape = env.image_shape

    num_epochs = args.num_epochs
    optimizer = optim.Adam(
        world_model.parameters(),
        **optimizer_kwargs,
    )
    train_losses = []
    test_losses = []
    os.makedirs("data/" + args.logdir + "/plots/", exist_ok=True)

    train_dataloader, test_dataloader, train_dataset, test_dataset = get_dataloader(
        "data/world_model_data/" + args.datafile + ".hdf5",
        args,
    )

    for i in tqdm(range(num_epochs)):
        print("Epoch: ", i)
        total_loss = 0
        total_train_steps = 0
        for data in train_dataloader:
            with torch.cuda.amp.autocast():
                obs, actions = data
                obs = obs.to(ptu.device).float()
                actions = actions.to(ptu.device).float()
                (
                    post,
                    prior,
                    post_dist,
                    prior_dist,
                    image_dist,
                    _,
                    _,
                    _,
                ) = world_model(obs, actions)
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
                total_loss += world_model_loss.item()
                total_train_steps += 1

            update_network(
                world_model, optimizer, world_model_loss, gradient_clip, scaler
            )
            scaler.update()
        train_loss = total_loss / total_train_steps
        train_losses.append(train_loss)
        print("Train Loss: ", train_loss)
        best_test_loss = np.inf
        with torch.no_grad():
            total_loss = 0
            total_test_steps = 0
            for data in test_dataloader:
                with torch.cuda.amp.autocast():
                    obs, actions = data
                    obs = obs.to(ptu.device).float()
                    actions = actions.to(ptu.device).float()
                    (
                        post,
                        prior,
                        post_dist,
                        prior_dist,
                        image_dist,
                        reward_dist,
                        pred_discount_dist,
                        _,
                    ) = world_model(obs, actions)
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
                    total_loss += world_model_loss.item()
                    total_test_steps += 1
            test_loss = total_loss / total_test_steps
            if test_loss <= best_test_loss:
                best_test_loss = test_loss
                os.makedirs("data/" + args.logdir + "/models/", exist_ok=True)
                torch.save(
                    world_model.state_dict(),
                    "data/" + args.logdir + "/models/world_model.pt",
                )
            print("Test Loss: ", test_loss)
            print()
            test_losses.append(test_loss)
        if i % 1 == 0:
            plt.plot(train_losses, label="Train Loss")
            plt.plot(test_losses, label="Test Loss")
            plt.title("Losses for World Model")
            plt.legend()
            plt.savefig("data/" + args.logdir + "/plots/losses.png")
            plt.clf()
            imagination_post_epoch_func(
                env, world_model, args, args.max_path_length, epoch=i
            )
            visualize_dataset_trajectory(
                env,
                train_dataset,
                world_model,
                args,
                args.max_path_length,
                i,
                tag="train",
            )
            visualize_dataset_trajectory(
                env,
                test_dataset,
                world_model,
                args,
                args.max_path_length,
                i,
                tag="test",
            )
    world_model.load_state_dict(
        torch.load("data/" + args.logdir + "/models/world_model.pt")
    )
    imagination_post_epoch_func(env, world_model, args, args.max_path_length, epoch=-1)
