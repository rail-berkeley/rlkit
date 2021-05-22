import copy

import numpy as np
import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
import torch
import torch.nn as nn
from railrl.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from railrl.envs.wrappers.predictive_wrapper_env import (
    RealNVPRewardShaperWrapper,
    RealNVPWrapper,
    ResidualRealNVPWrapper,
)
from railrl.launchers.launcher_util import run_experiment
from railrl.samplers.data_collector.path_collector import ObsDictPathCollector
from railrl.samplers.data_collector.step_collector import ObsDictStepCollector
from railrl.torch.networks import CNN, Flatten, MlpQfWithObsProcessor
from railrl.torch.sac.policies import MakeDeterministic, TanhGaussianPolicyAdapter
from railrl.torch.sac.sac import SACTrainer
from railrl.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)
from railrl.visualization.video import VideoSaveFunctionBullet

import rlkit.envs.primitives_make_env as primitives_make_env
import rlkit.torch.pytorch_util as rlkit_ptu
from rlkit.envs.primitives_wrappers import (
    DictObsWrapper,
    GetObservationWrapper,
    IgnoreLastAction,
)


def experiment(variant):
    ptu.set_gpu_mode(True, 0)
    rlkit_ptu.set_gpu_mode(True, 0)
    model_path = variant["model_path"]

    env_suite = variant.get("env_suite", "kitchen")
    env_name = variant["env"]
    env_kwargs = variant["env_kwargs"]
    base_env = GetObservationWrapper(
        IgnoreLastAction(
            DictObsWrapper(
                primitives_make_env.make_env(env_suite, env_name, env_kwargs)
            )
        )
    )
    action_dim = int(
        np.prod(base_env.action_space.shape)
    )  # add this as a bogus dim to the env as well
    base_env.cnn_input_key = "image"  # TODO(avi) clean this up
    base_env.fc_input_key = "state"
    img_width, img_height = base_env.imwidth, base_env.imheight
    num_channels = 3

    if variant["use_robot_state"]:
        observation_keys = (base_env.cnn_input_key, base_env.fc_input_key)
    else:
        observation_keys = (base_env.cnn_input_key,)

    if variant["use_residual_wrapper"]:
        expl_env = ResidualRealNVPWrapper(
            base_env,
            model_path,
            action_scale=variant["action_scale"],
            observation_keys=observation_keys,
        )
    elif variant["use_real_nvp_model"]:
        expl_env = RealNVPWrapper(
            base_env,
            model_path,
            action_scale=variant["action_scale"],
            observation_keys=observation_keys,
        )
    elif variant["use_reward_shaper"]:
        expl_env = RealNVPRewardShaperWrapper(
            base_env, model_path, observation_keys=observation_keys
        )
    else:
        expl_env = base_env

    eval_env = expl_env

    cnn_params = variant["cnn_params"]
    cnn_params.update(
        input_width=img_width,
        input_height=img_height,
        input_channels=num_channels,
    )

    if variant["use_robot_state"]:
        robot_state_obs_dim = expl_env.get_observation()[base_env.fc_input_key].shape[0]
        cnn_params.update(
            added_fc_input_size=robot_state_obs_dim,
            output_conv_channels=False,
            hidden_sizes=[400, 400],
            output_size=200,
            cnn_input_key=expl_env.cnn_input_key,
            fc_input_key=expl_env.fc_input_key,
        )
    else:
        cnn_params.update(
            added_fc_input_size=0,
            output_conv_channels=True,
            output_size=None,
            cnn_input_key=expl_env.cnn_input_key,
        )

    qf_cnn = CNN(**cnn_params)

    if variant["use_robot_state"]:
        qf_obs_processor = qf_cnn
        qf_cnn_output_size = qf_cnn.output_size
    else:
        qf_obs_processor = nn.Sequential(
            qf_cnn,
            Flatten(),
        )
        qf_cnn_output_size = qf_cnn.conv_output_flat_size

    qf_kwargs = copy.deepcopy(variant["qf_kwargs"])
    qf_kwargs["obs_processor"] = qf_obs_processor
    qf_kwargs["output_size"] = 1
    qf_kwargs["input_size"] = action_dim + qf_cnn_output_size
    qf1 = MlpQfWithObsProcessor(**qf_kwargs)
    qf2 = MlpQfWithObsProcessor(**qf_kwargs)

    target_qf_cnn = CNN(**cnn_params)

    if variant["use_robot_state"]:
        target_qf_obs_processor = target_qf_cnn
        target_qf_cnn_output_size = target_qf_cnn.output_size
    else:
        target_qf_obs_processor = nn.Sequential(
            target_qf_cnn,
            Flatten(),
        )
        target_qf_cnn_output_size = target_qf_cnn.conv_output_flat_size

    target_qf_kwargs = copy.deepcopy(variant["qf_kwargs"])
    target_qf_kwargs["obs_processor"] = target_qf_obs_processor
    target_qf_kwargs["output_size"] = 1
    target_qf_kwargs["input_size"] = action_dim + target_qf_cnn_output_size

    target_qf1 = MlpQfWithObsProcessor(**target_qf_kwargs)
    target_qf2 = MlpQfWithObsProcessor(**target_qf_kwargs)

    policy_cnn = CNN(**cnn_params)

    if variant["use_robot_state"]:
        policy_obs_processor = policy_cnn
        policy_cnn_output_size = policy_cnn.output_size
    else:
        policy_obs_processor = nn.Sequential(
            policy_cnn,
            Flatten(),
        )
        policy_cnn_output_size = policy_cnn.conv_output_flat_size

    policy = TanhGaussianPolicyAdapter(
        policy_obs_processor,
        policy_cnn_output_size,
        action_dim,
        **variant["policy_kwargs"]
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = ObsDictPathCollector(
        eval_env,
        eval_policy,
        observation_keys=observation_keys,
        **variant["eval_path_collector_kwargs"]
    )
    replay_buffer = ObsDictReplayBuffer(
        variant["replay_buffer_size"],
        expl_env,
        action_dim=action_dim,
        observation_keys=observation_keys,
    )

    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        use_robot_state=variant["use_robot_state"],
        **variant["trainer_kwargs"]
    )

    expl_path_collector = ObsDictStepCollector(
        expl_env,
        policy,
        observation_keys=observation_keys,
        **variant["expl_path_collector_kwargs"]
    )
    algorithm = TorchOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        use_robot_state=variant["use_robot_state"],
        **variant["algo_kwargs"]
    )
    print("TRAINING")
    video_func = VideoSaveFunctionBullet(variant)
    algorithm.post_train_funcs.append(video_func)
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        method="SAC+RealNVP",
        trainer_kwargs=dict(
            discount=0.99,
            # soft_target_tau=5e-3,
            # target_update_period=1,
            soft_target_tau=1.0,
            target_update_period=1000,
            policy_lr=3e-4,
            qf_lr=3e-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        algo_kwargs=dict(
            batch_size=256,
            # num_epochs=100,
            # num_eval_steps_per_epoch=50,
            # num_trains_per_train_loop=100,
            # num_expl_steps_per_train_loop=100,
            # min_num_steps_before_training=100,
            # max_path_length=10,
            max_path_length=60,
            num_epochs=2000,
            num_eval_steps_per_epoch=280 * 5,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=2500,
        ),
        cnn_params=dict(
            kernel_sizes=[3, 3],
            n_channels=[4, 4],
            strides=[1, 1],
            hidden_sizes=[32, 32],
            paddings=[1, 1],
            pool_type="max2d",
            pool_sizes=[2, 2],
            pool_strides=[2, 2],
            pool_paddings=[0, 0],
            image_augmentation=False,
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        dump_video_kwargs=dict(
            imsize=84,
            save_video_period=50,
        ),
        logger_config=dict(
            snapshot_mode="gap_and_last",
            snapshot_gap=50,
        ),
        dump_buffer_kwargs=dict(
            dump_buffer_period=10000,
        ),
        replay_buffer_size=int(2.5e6),
        expl_path_collector_kwargs=dict(),
        eval_path_collector_kwargs=dict(),
        shared_qf_conv=False,
        use_robot_state=False,
        use_real_nvp_model=True,
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
            imheight=84,
            imwidth=84,
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
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--sac-only", action="store_true", default=False)
    parser.add_argument("--use-reward-shaper", action="store_true", default=False)
    parser.add_argument("--use-residual-wrapper", action="store_true", default=False)
    parser.add_argument("--max-path-length", type=int, required=True)
    parser.add_argument("--target-object", type=str, default="")
    parser.add_argument("--task-reward", type=str, default="")

    parser.add_argument(
        "--obs", default="pixels", type=str, choices=("pixels", "pixels_debug")
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--reward-type", default="sparse", type=str)
    parser.add_argument("--action-scale", default=1.0, type=float)
    parser.add_argument("--use-img-aug", action="store_true", default=True)
    parser.add_argument("--use-robot-state", action="store_true", default=False)
    parser.add_argument(
        "--cnn", type=str, default="large", choices=("small", "large", "xlarge")
    )
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    variant["env"] = args.env
    variant["obs"] = args.obs
    variant["use_real_nvp_model"] = not args.sac_only
    variant["use_reward_shaper"] = args.use_reward_shaper
    variant["use_residual_wrapper"] = args.use_residual_wrapper

    if not variant["use_real_nvp_model"]:
        variant["method"] = "SAC"

    assert not (variant["use_reward_shaper"] and variant["use_real_nvp_model"])

    variant["model_path"] = args.model_path
    variant["target_object"] = args.target_object
    variant["reward_type"] = args.reward_type
    variant["task_reward"] = args.task_reward
    variant["algo_kwargs"]["max_path_length"] = args.max_path_length
    variant["action_scale"] = args.action_scale

    variant["use_robot_state"] = args.use_robot_state

    variant["cnn"] = args.cnn
    variant["seed"] = args.seed

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
        variant["qf_kwargs"].update(hidden_sizes=[1024, 512, 256])
        variant["policy_kwargs"].update(hidden_sizes=[1024, 512, 256])
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
        variant["qf_kwargs"].update(hidden_sizes=[1024, 512, 256])
        variant["policy_kwargs"].update(hidden_sizes=[1024, 512, 256])

    variant["cnn_params"]["image_augmentation"] = args.use_img_aug

    n_seeds = 3
    mode = "here_no_doodad"
    exp_prefix = "dev-{}".format(
        __file__.replace("/", "-").replace("_", "-").split(".")[0]
    )
    exp_prefix = "railrl-SAC-realNVP-{}-{}".format(args.env, args.obs)

    # n_seeds = 5
    # mode = 'ec2'
    # exp_prefix = 'railrl-bullet-sawyer-image-reach'

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
