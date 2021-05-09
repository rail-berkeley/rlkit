import copy
import pickle

import d4rl
import gym
import h5py
import numpy as np
import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
import torch.nn as nn
from railrl.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.buffer_save import BufferSaveFunction
from railrl.samplers.data_collector.path_collector import (
    CustomObsDictPathCollector,
    ObsDictPathCollector,
)
from railrl.torch.networks import CNN
from railrl.torch.sac.policies import ObservationConditionedRealNVP
from railrl.torch.sac.real_nvp import RealNVPTrainer
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from railrl.visualization.video import OnlinePathSaveFunction, VideoSaveFunctionBullet

import rlkit.envs.primitives_make_env as primitives_make_env
from rlkit.envs.primitives_wrappers import DictObsWrapper, IgnoreLastAction

DEFAULT_BUFFER = (
    "/media/avi/data/Work/data/widow250/"
    "Widow250MultiTaskGrasp-v0_grasping_two_both_trajs.npy"
)

from railrl.misc.buffer_load import load_data_from_npy


def experiment(variant):
    env_suite = variant.get("env_suite", "kitchen")
    env_name = variant["env"]
    env_kwargs = variant["env_kwargs"]
    expl_env = IgnoreLastAction(
        DictObsWrapper(primitives_make_env.make_env(env_suite, env_name, env_kwargs))
    )
    action_dim = int(
        np.prod(expl_env.action_space.shape)
    )  # add this as a bogus dim to the env as well

    expl_env.cnn_input_key = "image"  # TODO(avi) clean this up
    expl_env.fc_input_key = "state"
    eval_env = expl_env

    img_width, img_height = expl_env.imwidth, expl_env.imheight
    num_channels = 3

    print(action_dim)
    cnn_params = variant["cnn_params"]
    cnn_params.update(
        input_width=img_width,
        input_height=img_height,
        input_channels=num_channels,
    )
    if variant["use_robot_state"]:
        robot_state_obs_dim = expl_env.get_observation()[eval_env.fc_input_key].shape[0]
        cnn_params.update(
            added_fc_input_size=robot_state_obs_dim,
            output_conv_channels=False,
            hidden_sizes=[400, 400, 200],
            cnn_input_key=expl_env.cnn_input_key,
            fc_input_key=expl_env.fc_input_key,
        )
    else:
        cnn_params.update(
            added_fc_input_size=0,
            cnn_input_key=expl_env.cnn_input_key,
            output_conv_channels=False,
        )
    cnn = CNN(**cnn_params)

    assert variant["coupling_layers"] % 2 == 0
    flips = [False]
    for _ in range(variant["coupling_layers"] - 1):
        flips.append(not flips[-1])
    print("flips", flips)

    real_nvp_policy = ObservationConditionedRealNVP(
        flips,
        action_dim,
        obs_processor=cnn,
        ignore_observation=(not variant["observation_conditioning"]),
        use_atanh_preprocessing=variant["use_atanh"],
    )

    if variant["use_robot_state"]:
        observation_keys = (expl_env.cnn_input_key, expl_env.fc_input_key)
    else:
        observation_keys = (expl_env.cnn_input_key,)

    # Not actually used
    expl_path_collector = ObsDictPathCollector(
        expl_env,
        real_nvp_policy,
        observation_keys=observation_keys,
        **variant["eval_path_collector_kwargs"]
    )

    eval_path_collector = ObsDictPathCollector(
        eval_env,
        real_nvp_policy,
        observation_keys=observation_keys,
        **variant["eval_path_collector_kwargs"]
    )

    # replay_buffer, replay_buffer_validation = load_data_from_npy(
    #     variant, expl_env, observation_keys,
    #     limit_num_trajs=variant['limit_rb_num_trajs'])
    # with h5py.File(
    #     "/home/mdalal/research/spirl/data/kitchen-vision/kitchen-total-v0-vision-84.hdf5",
    #     "r",
    # ) as f:
    #     dataset = dict(
    #         observations=np.array(f["images"])
    #         .transpose(0, 3, 1, 2)
    #         .reshape(-1, 84 * 84 * 3),
    #         terminals=np.array(f["terminals"]),
    #         rewards=np.array(f["rewards"]),
    #         actions=np.array(f["actions"]),
    #     )
    with h5py.File(
        "/home/mdalal/research/spirl/data/kitchen-vision/kitchen-total-v0-vision-64.hdf5",
        "r",
    ) as f:
        dataset = dict(
            observations=np.array(f["images"])
            .transpose(0, 3, 1, 2)
            .reshape(-1, 64 * 64 * 3),
            terminals=np.array(f["terminals"]),
            rewards=np.array(f["rewards"]),
            actions=np.array(f["actions"]),
        )
    # add dummy dim of 0s to actions:
    dataset["actions"] = np.concatenate(
        (dataset["actions"], np.zeros((dataset["actions"].shape[0], 1))), axis=1
    )
    dataset = d4rl.qlearning_dataset(expl_env, dataset)

    SPLIT = dict(train=0.99, val=0.01, test=0.0)
    seq_end_idxs = np.where(dataset["terminals"])[0]
    start = 0
    seqs = []
    subseq_len = 280
    for end_idx in seq_end_idxs:
        if end_idx + 1 - start < subseq_len:
            continue  # skip too short demos
        seqs.append(
            dict(
                states=dataset["observations"][start : end_idx + 1],
            )
        )
        start = end_idx + 1
    n_seqs = len(seqs)

    train_start = 0
    train_end = int(SPLIT["train"] * n_seqs)
    val_start = int(SPLIT["train"] * n_seqs)
    val_end = int((SPLIT["train"] + SPLIT["val"]) * n_seqs)

    # TODO: implement train val splits
    replay_buffer = ObsDictReplayBuffer(
        max_size=dataset["actions"].shape[0],
        env=expl_env,
        observation_keys=("image",),
        action_dim=action_dim,
    )
    replay_buffer_validation = ObsDictReplayBuffer(
        max_size=dataset["actions"].shape[0],
        env=expl_env,
        observation_keys=("image",),
        action_dim=action_dim,
    )
    replay_buffer._terminals = dataset["terminals"].reshape(-1, 1)[
        train_start:train_end
    ]
    replay_buffer._rewards = dataset["rewards"].reshape(-1, 1)[train_start:train_end]
    replay_buffer._actions = dataset["actions"][train_start:train_end]
    replay_buffer._obs["image"] = dataset["observations"][train_start:train_end]
    replay_buffer._next_obs["image"] = dataset["next_observations"][
        train_start:train_end
    ]
    replay_buffer._size = train_end - train_start

    replay_buffer_validation._terminals = dataset["terminals"].reshape(-1, 1)[
        val_start:val_end
    ]
    replay_buffer_validation._rewards = dataset["rewards"].reshape(-1, 1)[
        val_start:val_end
    ]
    replay_buffer_validation._actions = dataset["actions"][val_start:val_end]
    replay_buffer_validation._obs["image"] = dataset["observations"][val_start:val_end]
    replay_buffer_validation._next_obs["image"] = dataset["next_observations"][
        val_start:val_end
    ]
    replay_buffer_validation._size = val_end - val_start
    trainer = RealNVPTrainer(
        env=eval_env,
        bijector=real_nvp_policy,
        use_robot_state=variant["use_robot_state"],
        **variant["trainer_kwargs"]
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        use_robot_state=variant["use_robot_state"],
        replay_buffer_validation=replay_buffer_validation,
        batch_rl=True,
        **variant["algo_kwargs"]
    )

    video_func = VideoSaveFunctionBullet(variant)
    # dump_buffer_func = BufferSaveFunction(variant)
    # algorithm.post_train_funcs.append(video_func)

    # online_data_func = OnlinePathSaveFunction(eval_env, variant)
    # algorithm.post_train_funcs.append(online_data_func)
    # algorithm.post_train_funcs.append(dump_buffer_func)

    algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        trainer_kwargs=dict(
            lr=1e-4,
        ),
        algo_kwargs=dict(
            batch_size=256,
            max_path_length=280,
            num_epochs=2000,
            num_eval_steps_per_epoch=300,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=10 * 1000,
            # max_path_length=10,
            # num_epochs=100,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=100,
            # min_num_steps_before_training=100,
        ),
        cnn_params=dict(
            kernel_sizes=[3, 3],
            n_channels=[4, 4],
            strides=[1, 1],
            hidden_sizes=[200, 200],
            paddings=[1, 1],
            pool_type="max2d",
            pool_sizes=[2, 2],
            pool_strides=[2, 2],
            pool_paddings=[0, 0],
            output_size=32,
            image_augmentation=False,
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        dump_video_kwargs=dict(
            imsize=48,
            save_video_period=50,
        ),
        logger_config=dict(
            snapshot_mode="gap_and_last",
            snapshot_gap=100,
        ),
        dump_buffer_kwargs=dict(
            dump_buffer_period=10000,
        ),
        replay_buffer_size=int(5e5),
        expl_path_collector_kwargs=dict(),
        eval_path_collector_kwargs=dict(),
        shared_qf_conv=False,
        use_robot_state=False,
        randomize_env=True,
        batch_rl=True,
        env_kwargs=dict(
            dense=False,
            image_obs=True,
            fixed_schema=False,
            action_scale=1,
            use_combined_action_space=True,
            proprioception=False,
            wrist_cam_concat_with_fixed_view=False,
            use_wrist_cam=False,
            normalize_proprioception_obs=True,
            use_workspace_limits=True,
            max_path_length=280,
            control_mode="joint_velocity",
            imwidth=64,
            imheight=64,
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=True,
                max_path_length=280,
                unflatten_images=False,
            ),
            image_kwargs=dict(),
        ),
        env_suite="kitchen",
    )

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--buffer", type=str, default=DEFAULT_BUFFER)
    parser.add_argument("--buffer-val", type=str, default="")
    parser.add_argument(
        "--obs", default="pixels", type=str, choices=("pixels", "pixels_debug")
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-obs-input", action="store_true", default=False)
    parser.add_argument("--eval-test-objects", action="store_true", default=False)
    parser.add_argument("--use-robot-state", action="store_true", default=False)
    parser.add_argument(
        "--cnn", type=str, default="large", choices=("small", "large", "xlarge")
    )
    parser.add_argument("--coupling-layers", type=int, default=4)
    parser.add_argument("--cnn-output-size", type=int, default=256)
    parser.add_argument("--use-img-aug", action="store_true", default=True)
    parser.add_argument("--no-use-atanh", action="store_true", default=False)
    parser.add_argument("--use-grad-clip", action="store_true", default=False)
    parser.add_argument("--grad-clip-threshold", type=float, default=50.0)
    parser.add_argument("--limit-rb-num-trajs", type=int, default=None)
    parser.add_argument("--seed", default=10, type=int)

    args = parser.parse_args()

    variant["env"] = args.env
    variant["obs"] = args.obs
    variant["buffer"] = args.buffer
    variant["buffer_validation"] = args.buffer_val

    variant["observation_conditioning"] = not args.no_obs_input
    variant["eval_test_objects"] = args.eval_test_objects
    variant["use_robot_state"] = args.use_robot_state
    variant["coupling_layers"] = args.coupling_layers

    variant["cnn"] = args.cnn

    if variant["cnn"] == "small":
        pass
    elif variant["cnn"] == "large":
        variant["cnn_params"].update(
            kernel_sizes=[3, 3, 3],
            n_channels=[16, 16, 16],
            strides=[1, 1, 1],
            hidden_sizes=[1024, 512],
            paddings=[1, 1, 1],
            pool_type="max2d",
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
        )
    elif variant["cnn"] == "xlarge":
        variant["cnn_params"].update(
            kernel_sizes=[3, 3, 3, 3],
            n_channels=[32, 32, 32, 32],
            strides=[1, 1, 1, 1],
            hidden_sizes=[1024, 512, 512],
            paddings=[1, 1, 1, 1],
            pool_type="max2d",
            pool_sizes=[2, 2, 1, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1, 1],
            pool_paddings=[0, 0, 0, 0],
        )

    variant["cnn_params"]["output_size"] = args.cnn_output_size
    variant["cnn_params"]["image_augmentation"] = args.use_img_aug
    variant["use_atanh"] = not args.no_use_atanh
    variant["trainer_kwargs"]["clip_gradients_by_norm"] = args.use_grad_clip
    variant["trainer_kwargs"][
        "clip_gradients_by_norm_threshold"
    ] = args.grad_clip_threshold
    variant["limit_rb_num_trajs"] = args.limit_rb_num_trajs
    variant["seed"] = args.seed

    n_seeds = 1
    mode = "here_no_doodad"
    exp_prefix = "railrl-realNVP-{}-{}".format(args.env, args.obs)

    # n_seeds = 5
    # mode = 'ec2'

    search_space = {
        "shared_qf_conv": [
            True,
            # False,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_name=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                gpu_id=args.gpu,
                unpack_variant=False,
            )
