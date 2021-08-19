"""Test VAL on Pybullet tasks.

The environment is available here:
https://github.com/anair13/bullet-manipulation-affordances

Data available for download:
https://drive.google.com/file/d/1SsVaQKZnY5UkuR78WrInp9XxTdKHbF0x/view

Pretrained VQVAE for download:

"""
import os
import sys

from rlkit.core import logger
from rlkit.testing import csv_util

import rlkit.util.hyperparameter as hyp
from rlkit.demos.source.encoder_dict_to_mdp_path_loader import EncoderDictToMDPPathLoader
from rlkit.launchers.experiments.ashvin.awac_rig import awac_rig_experiment, process_args
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.sac.policies import GaussianPolicy, GaussianMixturePolicy
from roboverse.envs.sawyer_rig_multiobj_v0 import SawyerRigMultiobjV0
from roboverse.envs.sawyer_rig_multiobj_tray_v0 import SawyerRigMultiobjTrayV0
from roboverse.envs.sawyer_rig_affordances_v0 import SawyerRigAffordancesV0
from rlkit.torch.networks import Clamp
from rlkit.torch.vae.vq_vae import VQ_VAE
from rlkit.torch.vae.vq_vae_trainer import VQ_VAETrainer
from rlkit.torch.grill.common import train_vqvae

demo_paths=[dict(path='sasha/affordances/combined/drawer_demos_0.pkl', obs_dict=True, is_demo=True),
            # dict(path='sasha/affordances/combined/drawer_demos_1.pkl', obs_dict=True, is_demo=True),
            # dict(path='sasha/affordances/combined/pnp_demos_0.pkl', obs_dict=True, is_demo=True),
            # dict(path='sasha/affordances/combined/tray_demos_0.pkl', obs_dict=True, is_demo=True),

            # dict(path='sasha/affordances/combined/drawer_demos_2.pkl', obs_dict=True, is_demo=True),
            # dict(path='sasha/affordances/combined/drawer_demos_3.pkl', obs_dict=True, is_demo=True),
            # dict(path='sasha/affordances/combined/pnp_demos_1.pkl', obs_dict=True, is_demo=True),
            # dict(path='sasha/affordances/combined/tray_demos_1.pkl', obs_dict=True, is_demo=True),

            # dict(path='sasha/affordances/combined/drawer_demos_4.pkl', obs_dict=True, is_demo=True),
            # dict(path='sasha/affordances/combined/drawer_demos_5.pkl', obs_dict=True, is_demo=True),
            # dict(path='sasha/affordances/combined/pnp_demos_2.pkl', obs_dict=True, is_demo=True),
            # dict(path='sasha/affordances/combined/tray_demos_2.pkl', obs_dict=True, is_demo=True),

            # dict(path='sasha/affordances/combined/drawer_demos_6.pkl', obs_dict=True, is_demo=True),
            # dict(path='sasha/affordances/combined/drawer_demos_7.pkl', obs_dict=True, is_demo=True),
            # dict(path='sasha/affordances/combined/pnp_demos_3.pkl', obs_dict=True, is_demo=True),
            # dict(path='sasha/affordances/combined/tray_demos_3.pkl', obs_dict=True, is_demo=True),
            ]

image_train_data = 'sasha/affordances/combined/combined_images.npy'
image_test_data = 'sasha/affordances/combined/combined_test_images.npy'

tray_goals = 'sasha/presampled_goals/affordances/combined/tray_goals.pkl'
pnp_goals = 'sasha/presampled_goals/affordances/combined/pnp_goals.pkl'

top_drawer_goals = 'sasha/presampled_goals/affordances/combined/top_drawer_goals.pkl'
bottom_drawer_goals = 'sasha/presampled_goals/affordances/combined/bottom_drawer_goals.pkl'

vqvae = "ashvin/valreproduce/combined1/run5/id0/best_vqvae.pt"

def main():
    variant = dict(
        imsize=48,
        env_kwargs=dict(
            test_env=True,
        ),
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
        ),

        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),

        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=1,
            use_automatic_entropy_tuning=False,
            alpha=0,

            bc_num_pretrain_steps=0,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=25000, #25000
            policy_weight_decay=1e-4,
            q_weight_decay=0,

            rl_weight=1.0,
            use_awr_update=True,
            use_reparam_update=False,
            compute_bc=True,
            reparam_weight=0.0,
            awr_weight=1.0,
            bc_weight=0.0,

            reward_transform_kwargs=None,
            terminal_transform_kwargs=None,
        ),

        max_path_length=65, #50
        algo_kwargs=dict(
            batch_size=1024, #1024
            num_epochs=1001, #1001
            num_eval_steps_per_epoch=1000, #1000
            num_expl_steps_per_train_loop=1000, #1000
            num_trains_per_train_loop=1000, #1000
            min_num_steps_before_training=4000, #4000
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.6,
            fraction_distribution_context=0.1,
            max_size=int(5E5),
        ),
        demo_replay_buffer_kwargs=dict(
            fraction_future_context=0.6,
            fraction_distribution_context=0.1,
        ),
        reward_kwargs=dict(
            reward_type='sparse',
            epsilon=1.0,
        ),

        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        save_video=False,
        save_video_kwargs=dict(
            save_video_period=25,
            pad_color=0,
        ),

        reset_keys_map=dict(
            image_observation="initial_latent_state"
        ),
        pretrained_vae_path=vqvae,

        path_loader_class=EncoderDictToMDPPathLoader,
        path_loader_kwargs=dict(
            delete_after_loading=True,
            recompute_reward=True,
            demo_paths=demo_paths,
        ),

        renderer_kwargs=dict(
            create_image_format='HWC',
            output_image_format='CWH',
            flatten_image=True,
            width=48,
            height=48,
        ),

        add_env_demos=False,
        add_env_offpolicy_data=False,

        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,

        evaluation_goal_sampling_mode="presampled_images",
        exploration_goal_sampling_mode="presample_latents",

        train_vae_kwargs=dict(
            imsize=48,
            beta=1,
            beta_schedule_kwargs=dict(
                x_values=(0, 250),
                y_values=(0, 100),
            ),
            num_epochs=1501, #1501
            embedding_dim=5,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            use_linear_dynamics=False,
            generate_vae_dataset_kwargs=dict(
                N=1000,
                n_random_steps=2,
                test_p=.9,
                dataset_path={'train': image_train_data,
                              'test': image_test_data,
                              },
                augment_data=False,
                use_cached=False,
                show=False,
                oracle_dataset=False,
                oracle_dataset_using_set_to_goal=False,
                non_presampled_goal_img_is_garbage=False,
                delete_after_loading=True,
                random_rollout_data=True,
                random_rollout_data_set_to_goal=True,
                conditional_vae_dataset=True,
                save_trajectories=False,
                enviorment_dataset=False,
                tag="ccrig_tuning_orig_network",
            ),
            vae_trainer_class=VQ_VAETrainer,
            vae_class=VQ_VAE,
            vae_kwargs=dict(
                input_channels=3,
                imsize=48,
            ),
            algo_kwargs=dict(
                key_to_reconstruct='x_t',
                start_skew_epoch=5000,
                is_auto_encoder=False,
                batch_size=128,
                lr=1e-3,
                skew_config=dict(
                    method='vae_prob',
                    power=0,
                ),
                weight_decay=0.0,
                skew_dataset=False,
                priority_function_kwargs=dict(
                    decoder_distribution='gaussian_identity_variance',
                    sampling_method='importance_sampling',
                    num_latents_to_sample=10,
                ),
                use_parallel_dataloading=False,
            ),

            save_period=50,
        ),
        train_model_func=train_vqvae,
        presampled_goal_kwargs=dict(
            eval_goals='', #HERE
            expl_goals='',
        ),
        launcher_config=dict(
            unpack_variant=True,
            region='us-west-1', #HERE
        ),
    )

    search_space = {
        "seed": range(2),

        'env_type': ['top_drawer', 'bottom_drawer', 'tray', 'pnp'],
        'reward_kwargs.epsilon': [3.5, 4.0], #3.5, 4.0, 4.5, 5.0, 5.5, 6.0

        'trainer_kwargs.beta': [0.3],
        # 'num_pybullet_objects':[None],
        'policy_kwargs.min_log_std': [-6],
        'trainer_kwargs.awr_weight': [1.0],
        'trainer_kwargs.awr_use_mle_for_vf': [True, ],
        'trainer_kwargs.awr_sample_actions': [False, ],
        'trainer_kwargs.clip_score': [2, ],
        'trainer_kwargs.awr_min_q': [True, ],
        'trainer_kwargs.reward_transform_kwargs': [None, ],
        'trainer_kwargs.terminal_transform_kwargs': [dict(m=0, b=0),],
        'qf_kwargs.output_activation': [Clamp(max=0)],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        env_type = variant['env_type']
        eval_goals = 'sasha/presampled_goals/affordances/combined/{0}_goals.pkl'.format(env_type)
        variant['presampled_goal_kwargs']['eval_goals'] = eval_goals

        if env_type in ['top_drawer', 'bottom_drawer']:
            variant['env_class'] = SawyerRigAffordancesV0
            variant['env_kwargs']['env_type'] = env_type
        if env_type == 'tray':
            variant['env_class'] = SawyerRigMultiobjTrayV0
        if env_type == 'pnp':
            variant['env_class'] = SawyerRigMultiobjV0

        variants.append(variant)

    run_variants(awac_rig_experiment, variants, run_id=0, process_args_fn=process_args) #HERE


def test_val_full_online():
    cmd = "python examples/val/mini_pretrained1.py --1 --local --gpu --run 0 --seed 0 --debug"
    sys.argv = cmd.split(" ")[1:]
    main()

    # check if offline training results matches
    reference_csv = "tests/regression/val/id0_pretrained/pretrain_q.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "pretrain_q.csv")
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["trainer/batch", "trainer/Advantage Score Max", "trainer/Q1 Predictions Mean", "trainer/replay_buffer_len"]
    csv_util.check_equal(reference, output, keys)

    # check if online training results match
    reference_csv = "tests/regression/val/id0_pretrained/progress.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "progress.csv")
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["epoch", "eval/Actions Mean", "expl/Actions Mean", "eval/Average Returns", "expl/Average Returns", "trainer/Advantage Score Max", "trainer/Q1 Predictions Mean", "trainer/replay_buffer_len"]
    csv_util.check_equal(reference, output, keys)

if __name__ == "__main__":
    test_val_full_online()
