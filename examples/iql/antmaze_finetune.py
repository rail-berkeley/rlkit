"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.hdf5_path_loader import HDF5PathLoader
from rlkit.launchers.experiments.awac.finetune_rl import experiment, process_args

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment

from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.torch.sac.iql_trainer import IQLTrainer

# import d4rl.gym_mujoco
import d4rl

def main():
    variant = dict(
        algo_kwargs=dict(
            start_epoch=-1000, # offline epochs
            num_epochs=1001, # online epochs
            batch_size=256,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
        ),
        max_path_length=1000,
        replay_buffer_size=int(2E6),
        layer_size=256,
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, ],
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
            # num_gaussians=1,
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256, ],
        ),

        algorithm="SAC",
        version="normal",
        collection_mode='batch',
        trainer_class=IQLTrainer,
        trainer_kwargs=dict(
            discount=0.99,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,

            policy_weight_decay=1e-4,
            q_weight_decay=0,

            reward_transform_kwargs=None,
            terminal_transform_kwargs=None,

            beta=1,
            quantile=0.9,
            clip_score=100,
        ),
        launcher_config=dict(
            num_exps_per_instance=1,
            region='us-west-2',
        ),

        path_loader_class=HDF5PathLoader,
        path_loader_kwargs=dict(),
        add_env_demos=False,
        add_env_offpolicy_data=False,

        # logger_variant=dict(
        #     tensorboard=True,
        # ),
        load_demos=False,
        load_env_dataset_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,
        # save_pretrained_algorithm=True,
        # snapshot_mode="all",
    )

    search_space = {
        'normalize_env': [False],
        'use_validation_buffer': [False], # changed this line, added
        'policy_kwargs.std': [None, ],
        'env_id': [
            "antmaze-umaze-v0", "antmaze-umaze-diverse-v0", "antmaze-medium-play-v0",
            "antmaze-medium-diverse-v0", "antmaze-large-diverse-v0", "antmaze-large-play-v0",
        ],
        'trainer_kwargs.beta': [0.1, ],
        'policy_kwargs.std_architecture': ["values", ],
        'trainer_kwargs.q_weight_decay': [0, ],
        'trainer_kwargs.reward_transform_kwargs': [dict(m=1, b=-1), ],
        'seedid': range(3),
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    use_gpu = True

    n_seeds = 1
    mode = 'here_no_doodad'
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

if __name__ == "__main__":
    main()
