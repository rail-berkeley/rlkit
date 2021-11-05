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
from tqdm import tqdm

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.primitives_make_env import make_env
from rlkit.torch.model_based.dreamer.world_models import WorldModel

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@torch.no_grad()
def imagination_post_epoch_func(env, world_model, args, max_path_length, epoch):
    print("Generating Imagination Reconstructions: ")
    with torch.cuda.amp.autocast():
        null_state = world_model.initial(4)
        null_acts = ptu.zeros((4, 4))
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
    def __init__(self, observations, actions):
        """
        :param observations: (np.ndarray)
        :param actions: (np.ndarray)
        :param max_path_length: (int)
        """
        self.observations, self.actions = observations, actions

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


def get_dataloader(filename, train_test_split=0.8, max_path_length=1):
    """
    :param filename: (str)
    :param train_test_split: (float)
    :param max_path_length: (int)
    :return: (tuple)
    """
    args = get_args()
    # data = np.load(filename, allow_pickle=True).item()
    with h5py.File(filename, "r") as f:
        observations = np.array(f["observations"][:])
        actions = np.array(f["actions"][:])
    # observations, actions = preprocess_data(observations, actions, max_path_length=max_path_length)
    observations, actions = torch.from_numpy(observations), torch.from_numpy(actions)

    num_train_datapoints = int(observations.shape[0] * train_test_split)
    train_dataset = NumpyDataset(
        observations[:num_train_datapoints], actions[:num_train_datapoints]
    )
    test_dataset = NumpyDataset(
        observations[num_train_datapoints:], actions[num_train_datapoints:]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    args = get_args()
    ptu.device = torch.device("cuda")
    scaler = torch.cuda.amp.GradScaler()
    world_model = WorldModel(
        model_hidden_size=400,
        stochastic_state_size=50,
        deterministic_state_size=200,
        embedding_size=1024,
        rssm_hidden_size=200,
        reward_num_layers=2,
        pred_discount_num_layers=3,
        gru_layer_norm=True,
        std_act="sigmoid2",
        action_dim=4,
        image_shape=(3, 64, 64),
    ).to(ptu.device)
    gradient_clip = 100
    image_shape = (3, 64, 64)

    num_epochs = args.num_epochs
    optimizer = optim.Adam(
        world_model.parameters(),
        lr=3e-4,
        eps=1e-5,
        weight_decay=0.0,
    )
    train_losses = []
    test_losses = []
    os.makedirs("data/" + args.logdir + "/plots/", exist_ok=True)

    train_dataloader, test_dataloader = get_dataloader(
        "data/world_model_data/" + args.datafile + ".hdf5",
        train_test_split=0.8,
        max_path_length=args.max_path_length,
    )

    env_kwargs = dict(
        control_mode="end_effector",
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
        collect_primitives_info=True,
        include_phase_variable=True,
        render_intermediate_obs_to_info=True,
    )
    env_suite = "metaworld"
    env_name = "reach-v2"
    env = make_env(env_suite, env_name, env_kwargs)

    for i in tqdm(range(num_epochs)):
        imagination_post_epoch_func(
            env, world_model, args, args.max_path_length, epoch=i
        )
        total_loss = 0
        total_train_steps = 0
        print("Epoch: ", i)
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
                    forward_kl=False,
                    free_nats=1.0,
                    transition_loss_scale=0.8,
                    kl_loss_scale=0.0,
                    image_loss_scale=1.0,
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
                        forward_kl=False,
                        free_nats=1.0,
                        transition_loss_scale=0.8,
                        kl_loss_scale=0.0,
                        image_loss_scale=1.0,
                    )
                    total_loss += world_model_loss.item()
                    total_test_steps += 1
                    if total_loss <= best_test_loss:
                        best_test_loss = total_loss

            test_loss = total_loss / total_test_steps
            if test_loss <= best_test_loss:
                os.makedirs("data/" + args.logdir + "/models/", exist_ok=True)
                torch.save(
                    world_model.state_dict(),
                    "data/" + args.logdir + "/models/world_model.pt",
                )
            print("Test Loss: ", test_loss)
            print()
            test_losses.append(test_loss)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.title("Losses for World Model")
        plt.legend()
        plt.savefig("data/" + args.logdir + "/plots/losses.png")
        plt.clf()
    world_model.load_state_dict(
        torch.load("data/" + args.logdir + "/models/world_model.pt")
    )
    imagination_post_epoch_func(env, world_model, args, args.max_path_length, epoch=-1)
