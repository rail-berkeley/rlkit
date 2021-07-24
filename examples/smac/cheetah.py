from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.smac.launcher import smac_experiment
from rlkit.torch.smac.base_config import DEFAULT_CONFIG
import rlkit.util.hyperparameter as hyp


# @click.command()
# @click.option('--debug', is_flag=True, default=False)
# @click.option('--dry', is_flag=True, default=False)
# @click.option('--suffix', default=None)
# @click.option('--nseeds', default=1)
# @click.option('--mode', default='here_no_doodad')
# def main(debug, dry, suffix, nseeds, mode):
def main():
    debug = True
    dry = False
    mode = 'here_no_doodad'
    suffix = ''
    nseeds = 1
    gpu=True

    path_parts = __file__.split('/')
    suffix = '' if suffix is None else '--{}'.format(suffix)
    exp_name = 'pearl-awac-{}--{}{}'.format(
        path_parts[-2].replace('_', '-'),
        path_parts[-1].split('.')[0].replace('_', '-'),
        suffix,
    )

    if debug or dry:
        exp_name = 'dev--' + exp_name
        mode = 'here_no_doodad'
        nseeds = 1

    print(exp_name)

    variant = DEFAULT_CONFIG.copy()
    variant["env_name"] = "cheetah-vel"
    search_space = {
        'load_buffer_kwargs.pretrain_buffer_path': [
            "results/.../extra_snapshot_itr100.cpkl"  # TODO: update to point to correct file
        ],
        'saved_tasks_path': [
            "examples/smac/cheetah_tasks.joblib",  # TODO: update to point to correct file
        ],
        'load_macaw_buffer_kwargs.rl_buffer_start_end_idxs': [
            [(0, 1200)],
        ],
        'load_macaw_buffer_kwargs.encoder_buffer_start_end_idxs': [
            [(-400, None)],
        ],
        'load_macaw_buffer_kwargs.encoder_buffer_matches_rl_buffer': [
            False,
        ],
        'algo_kwargs.use_rl_buffer_for_enc_buffer': [
            False,
        ],
        'algo_kwargs.train_encoder_decoder_in_unsupervised_phase': [
            False,
        ],
        'algo_kwargs.freeze_encoder_buffer_in_unsupervised_phase': [
            False,
        ],
        'algo_kwargs.use_encoder_snapshot_for_reward_pred_in_unsupervised_phase': [
            True,
        ],
        'pretrain_offline_algo_kwargs.logging_period': [
            25000,
        ],
        'algo_kwargs.num_iterations': [
            51,
        ],
        'seed': list(range(nseeds)),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant['exp_id'] = exp_id
        run_experiment(
            smac_experiment,
            unpack_variant=True,
            exp_prefix=exp_name,
            mode=mode,
            variant=variant,
            use_gpu=gpu,
        )

    print(exp_name)




if __name__ == "__main__":
    main()

