"""
PEARL Experiment
"""

import pickle
import click
from pathlib import Path

from rlkit.launchers.launcher_util import load_pyhocon_configs
import rlkit.pythonplusplus as ppp
from rlkit.torch.pearl.cql_launcher import pearl_cql_experiment
import rlkit.misc.hyperparameter as hyp
from rlkit.torch.pearl.sac_launcher import pearl_sac_experiment
from rlkit.torch.pearl.awac_launcher import pearl_awac_experiment


name_to_exp = {
    'CQL': pearl_cql_experiment,
    'AWAC': pearl_awac_experiment,
    'SAC': pearl_sac_experiment,
}


@click.command()
@click.option('--debug', is_flag=True, default=False)
@click.option('--dry', is_flag=True, default=False)
@click.option('--suffix', default=None)
@click.option('--nseeds', default=1)
@click.option('--mode', default='local')
@click.option('--olddd', is_flag=True, default=False)
def main(debug, dry, suffix, nseeds, mode, olddd):
    gpu = True

    base_dir = Path(__file__).parent.parent

    path_parts = __file__.split('/')
    suffix = '' if suffix is None else '--{}'.format(suffix)
    exp_name = 'pearl-awac-{}--{}{}'.format(
        path_parts[-2].replace('_', '-'),
        path_parts[-1].split('.')[0].replace('_', '-'),
        suffix,
    )

    if debug or dry:
        exp_name = 'dev--' + exp_name
        mode = 'local'
        nseeds = 1

    if dry:
        mode = 'here_no_doodad'

    print(exp_name)

    def postprocess_config(variant):
        encoder_buffer_mode = variant['tags']['encoder_buffer_mode']
        if encoder_buffer_mode == 'frozen':
            encoder_buffer_matches_rl_buffer = False
            freeze_encoder_buffer_in_unsupervised_phase = True
            clear_encoder_buffer_before_every_update = False
        elif encoder_buffer_mode == 'keep_latest_exploration_only':
            encoder_buffer_matches_rl_buffer = False
            freeze_encoder_buffer_in_unsupervised_phase = False
            clear_encoder_buffer_before_every_update = True
        elif encoder_buffer_mode == 'keep_all_exploration':
            encoder_buffer_matches_rl_buffer = False
            freeze_encoder_buffer_in_unsupervised_phase = False
            clear_encoder_buffer_before_every_update = False
        elif encoder_buffer_mode == 'match_rl':
            encoder_buffer_matches_rl_buffer = True
            freeze_encoder_buffer_in_unsupervised_phase = False
            clear_encoder_buffer_before_every_update = False
        else:
            raise ValueError(encoder_buffer_mode)
        variant['algo_kwargs']['encoder_buffer_matches_rl_buffer'] = encoder_buffer_matches_rl_buffer
        variant['algo_kwargs']['freeze_encoder_buffer_in_unsupervised_phase'] = freeze_encoder_buffer_in_unsupervised_phase
        variant['algo_kwargs']['clear_encoder_buffer_before_every_update'] = clear_encoder_buffer_before_every_update
        return variant

    def run_sweep(search_space, variant):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            variant['expd_id'] = exp_id
            run_experiment(
                name_to_exp[variant['tags']['method']],
                unpack_variant=True,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                time_in_mins=3 * 24 * 60 - 1,
                use_gpu=gpu,
                mount_point=None,
            )

    configs = [
        base_dir / 'configs/default_sac.conf',
        base_dir / 'configs/half_cheetah_130_offline.conf',
        ]
    if debug:
        configs.append(base_dir / 'configs/debug.conf')
    variant = ppp.recursive_to_dict(load_pyhocon_configs(configs))
    tasks = pickle.load(open('/home/vitchyr/mnt2/log2/demos/half_cheetah_vel_130/half_cheetah_vel_130_tasks.pkl', 'rb'))
    search_space = {
        'seed': list(range(nseeds)),
        'trainer_kwargs.train_context_decoder': [
            True,
        ],
        'trainer_kwargs.backprop_q_loss_into_encoder': [
            False,
            True,
        ],
        'train_task_idxs': [
            list(range(100)),
        ],
        'eval_task_idxs': [
            list(range(100, 130))
        ],
        'env_params.presampled_tasks': [
            tasks,
        ],
        'algo_kwargs.num_iterations_with_reward_supervision': [
            None,
        ],
        'algo_kwargs.exploration_resample_latent_period': [
            1,
        ],
        'tags.encoder_buffer_mode': [
            'keep_latest_exploration_only',
            'match_rl',
        ],
    }

    run_sweep(search_space, variant)


if __name__ == "__main__":
    main()

