import os

import matplotlib
import numpy as np
import torch
import torch.optim as optim
from arguments import get_args
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rlkit.envs.primitives_make_env import make_env
from rlkit.torch.model_based.dreamer.mlp import Mlp

matplotlib.use("Agg")
from matplotlib import pyplot as plt


class NumpyDataset(Dataset):
    def __init__(self, inputs, outputs, input_subselect="all"):
        """
        :param inputs: (np.ndarray)
        :param outputs: (np.ndarray)
        """
        self.inputs = inputs
        self.outputs = outputs
        self.input_subselect = input_subselect

    def __len__(self):
        """
        :return: (int)
        """
        return self.inputs.shape[0]

    def __getitem__(self, i):
        """
        :param i: (int)
        :return: (dict)
        """
        if self.input_subselect == "ee":
            inp = self.inputs[i, 20:]
        elif self.input_subselect == "all":
            inp = self.inputs[i]
        return inp, self.outputs[i]


def get_dataloaders(filename, num_primitives, train_test_split=0.8):
    """
    :param filename: (str)
    :param num_primitives: (int)
    :param train_test_split: (float)
    :return: (tuple)
    """
    args = get_args()
    data = np.load(filename, allow_pickle=True).item()
    data["inputs"] = [
        torch.Tensor(np.array(data["inputs"][i])) for i in range(num_primitives)
    ]
    data["inputs"] = [
        data["inputs"][i].reshape(-1, data["inputs"][i].shape[-1])
        for i in range(num_primitives)
    ]

    data["actions"] = [
        torch.Tensor(np.array(data["actions"][i])) for i in range(num_primitives)
    ]
    data["actions"] = [
        data["actions"][i].reshape(-1, data["actions"][i].shape[-1])
        for i in range(num_primitives)
    ]

    num_train_datapoints = [
        int(len(data["actions"][i]) * train_test_split) for i in range(num_primitives)
    ]
    train_inputs = [
        data["inputs"][i][: num_train_datapoints[i]] for i in range(num_primitives)
    ]
    train_outputs = [
        data["actions"][i][: num_train_datapoints[i]] for i in range(num_primitives)
    ]

    test_inputs = [
        data["inputs"][i][num_train_datapoints[i] :] for i in range(num_primitives)
    ]
    test_outputs = [
        data["actions"][i][num_train_datapoints[i] :] for i in range(num_primitives)
    ]

    train_dataloaders = [
        DataLoader(
            NumpyDataset(inputs, outputs, args.input_subselect),
            batch_size=args.batch_size,
            shuffle=True,
        )
        for inputs, outputs in zip(train_inputs, train_outputs)
    ]

    test_dataloaders = [
        DataLoader(
            NumpyDataset(inputs, outputs, args.input_subselect),
            batch_size=args.batch_size,
            shuffle=True,
        )
        for inputs, outputs in zip(test_inputs, test_outputs)
    ]
    return data, train_dataloaders, test_dataloaders


if __name__ == "__main__":
    env_kwargs = dict(
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
            use_image_obs=False,
            max_path_length=5,
            unflatten_images=False,
        ),
        image_kwargs=dict(imwidth=64, imheight=64),
    )
    env_suite = "metaworld"
    env_name = "reach-v2"
    env = make_env(env_suite, env_name, env_kwargs)
    num_primitives = env.num_primitives

    args = get_args()

    data, train_dataloaders, test_dataloaders = get_dataloaders(
        "data/primitive_data/" + args.datafile + ".npy",
        num_primitives,
        train_test_split=0.8,
    )

    primitives = []
    for i in range(num_primitives):
        input_size = data["inputs"][i].shape[-1]
        if args.input_subselect == "ee":
            input_size = input_size - 20
        primitives.append(
            Mlp(
                hidden_sizes=args.hidden_sizes,
                output_size=data["actions"][i].shape[1],
                input_size=input_size,
                hidden_activation=torch.nn.functional.relu,
            ).cuda()
        )

    for i in range(num_primitives):
        print(data["actions"][i].shape)
        print(
            data["actions"][i].mean().item(),
            data["actions"][i].std().item(),
            data["actions"][i].min().item(),
            data["actions"][i].max().item(),
        )
        print(
            data["inputs"][i].mean().item(),
            data["inputs"][i].std().item(),
            data["inputs"][i].min().item(),
            data["inputs"][i].max().item(),
        )

    criterion = nn.MSELoss()
    optimizers = [optim.Adam(p.parameters(), lr=args.lr) for p in primitives]
    train_losses = [[] for i in range(num_primitives)]
    test_losses = [[] for i in range(num_primitives)]
    num_epochs = args.num_epochs

    os.makedirs("data/" + args.logdir + "/plots/", exist_ok=True)
    os.makedirs("data/" + args.logdir + "/models/", exist_ok=True)
    best_test_losses = 100 * np.ones(num_primitives)
    for i in tqdm(range(num_epochs)):
        print(i)
        for primitive, (train_dataloader, test_dataloader, p, optimizer) in enumerate(
            zip(train_dataloaders, test_dataloaders, primitives, optimizers)
        ):
            total_loss = 0
            total_train_steps = 0
            for data in train_dataloader:
                optimizer.zero_grad()
                inputs, outputs = data
                inputs = inputs.cuda()
                outputs = outputs.cuda()
                predicted_outputs = p(inputs)
                loss = criterion(outputs, predicted_outputs)
                total_loss += loss.item()
                total_train_steps += 1
                loss.backward()
                optimizer.step()
            print("Train Loss {}: ".format(primitive), total_loss / total_train_steps)
            train_losses[primitive].append(total_loss / total_train_steps)

            with torch.no_grad():
                total_loss = 0
                total_test_steps = 0
                for data in test_dataloader:
                    inputs, outputs = data
                    inputs = inputs.cuda()
                    outputs = outputs.cuda()
                    predicted_outputs = p(inputs)
                    loss = criterion(outputs, predicted_outputs)
                    total_loss += loss.item()
                    total_test_steps += 1

                test_loss = total_loss / total_test_steps
                best_test_losses[primitive] = min(
                    test_loss, best_test_losses[primitive]
                )
                if test_loss <= best_test_losses[primitive]:
                    torch.save(
                        p.state_dict(),
                        "data/"
                        + args.logdir
                        + "/models/primitive_{}.pt".format(primitive),
                    )
                print("Test MSE {}: ".format(primitive), test_loss)
                test_losses[primitive].append(test_loss)
                print()
        if i % 10 == 0:
            for k in range(num_primitives):
                plt.plot(train_losses[k], label="Train Loss {}".format(k))
                plt.plot(test_losses[k], label="Test Loss {}".format(k))
                plt.title("Losses for primitive {}".format(k))
                plt.legend()
                plt.savefig("data/" + args.logdir + "/plots/losses_{}.png".format(k))
                plt.clf()
