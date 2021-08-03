import matplotlib
import numpy as np
import torch
from arguments import get_args
from clone_primitives import get_dataloaders
from torch import nn

from rlkit.envs.primitives_make_env import make_env
from rlkit.torch.model_based.dreamer.mlp import Mlp

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def load_primitives_and_dataloaders(datapath):
    args = get_args()
    data, train_dataloaders, test_dataloaders = get_dataloaders(
        datapath, num_primitives, train_test_split=0.8
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
            )
        )

    for i in range(num_primitives):
        primitives[i].load_state_dict(
            torch.load("data/" + args.logdir + "/models/primitive_{}.pt".format(i))
        )
        primitives[i].eval().cpu()

    return primitives, train_dataloaders, test_dataloaders


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
        collect_primitives_info=True,
        include_phase_variable=True,
    )
    env_suite = "metaworld"
    env_name = "reach-v2"
    env = make_env(env_suite, env_name, env_kwargs)
    num_primitives = env.num_primitives

    args = get_args()

    primitives, train_dataloaders, test_dataloaders = load_primitives_and_dataloaders(
        "data/primitive_data/" + args.datafile + ".npy",
    )

    criterion = nn.MSELoss()
    with torch.no_grad():
        for primitive, (test_dataloader, p) in enumerate(
            zip(test_dataloaders, primitives)
        ):
            total_loss = 0
            total_test_steps = 0
            for data in test_dataloader:
                inputs, outputs = data
                predicted_outputs = p(inputs)
                loss = criterion(outputs, predicted_outputs)
                total_loss += loss.item()
                total_test_steps += 1
            print("Test MSE {}: ".format(primitive), total_loss / total_test_steps)
            print()

    print("Sequential Test")
    action_errors = []
    state_errors = []
    with torch.no_grad():
        for i in range(100):
            a = env.action_space.sample()
            a[1] = 100
            obs = env.reset()
            o, r, d, i = env.step(
                a,
            )
            true_actions = np.array(i["actions"])
            true_states = np.array(i["robot-states"])
            env_kwargs["learned_primitives"] = primitives
            env_kwargs["use_learned_primitives"] = True
            env = make_env(env_suite, env_name, env_kwargs)
            obs = env.reset()
            o, r, d, i = env.step(
                a,
            )
            predicted_actions = np.array(i["actions"])

            action_errors.append(
                np.square(np.linalg.norm(true_actions - predicted_actions, axis=1))
            )
            predicted_states = np.array(i["robot-states"])
            state_errors.append(
                np.square(np.linalg.norm(true_states - predicted_states, axis=1))
            )
    action_errors = np.array(action_errors)
    action_errors = np.mean(action_errors, axis=0)
    plt.plot(action_errors)
    plt.title("Average Sequential Mean Squared Error (Actions)")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.savefig("data/" + args.logdir + "/plots/action.png")
    plt.clf()
    state_errors = np.array(state_errors)
    state_errors = np.mean(state_errors, axis=0)
    plt.plot(state_errors)
    plt.title("Average Sequential Mean Squared Error (States)")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.savefig("data/" + args.logdir + "/plots/state.png")
