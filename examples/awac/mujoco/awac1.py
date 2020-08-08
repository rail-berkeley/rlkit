from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.launchers.experiments.awac.awac_rl import experiment, process_args

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment

from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.torch.networks import Clamp

if __name__ == "__main__":
    variant = dict(
        num_epochs=501,
        num_eval_steps_per_epoch=1000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=1024,
        replay_buffer_size=int(1E6),
        layer_size=256,
        num_layers=2,
        algorithm="AWAC",
        version="normal",
        collection_mode='batch',

        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256] * 4,
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256]
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=1,
            alpha=0,
            use_automatic_entropy_tuning=False,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=50000,
            policy_weight_decay=1e-4,
            train_bc_on_rl_buffer=False,
            buffer_policy_sample_actions=False,

            reparam_weight=0.0,
            awr_weight=1.0,
            bc_weight=0.0,
            compute_bc=False,
            awr_use_mle_for_vf=False,
            awr_sample_actions=False,
            awr_min_q=True,
        ),
        path_loader_kwargs=dict(
            demo_paths=[  # these can be loaded in awac_rl.py per env
                # dict(
                #     path='demos/ant_action_noise_15.npy',
                #     obs_dict=False,
                #     is_demo=True,
                #     train_split=.9,
                # ),
            ],
        ),
        path_loader_class=DictToMDPPathLoader,

        pretrain_rl=True,
        use_validation_buffer=True,
        add_env_demos=True,
        add_env_offpolicy_data=True,
        load_demos=True,
    )

    search_space = {
        'trainer_kwargs.beta':[2, ],
        'train_rl':[True],
        'pretrain_rl':[True],
        'pretrain_policy':[False],
        'env_id': ['HalfCheetah-v2', 'Ant-v2', 'Walker2d-v2', ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    # n_seeds = 3
    # mode = 'gcp'
    # exp_prefix = 'skew-fit-pickup-reference-post-refactor'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=use_gpu,
                snapshot_gap=200,
                snapshot_mode='gap_and_last',
                num_exps_per_instance=3,
                gcp_kwargs=dict(
                    zone='us-west1-b',
                ),

            )

    # variants = []
    # for variant in sweeper.iterate_hyperparameters():
    #     variants.append(variant)

    # run_variants(experiment, variants, process_args)
