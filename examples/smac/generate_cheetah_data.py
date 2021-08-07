"""
PEARL Experiment
"""

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.smac.base_config import DEFAULT_PEARL_CONFIG

from rlkit.torch.smac.pearl_launcher import pearl_experiment
from rlkit.util.io import load_local_or_remote_file


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
    gpu = True

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

    if dry:
        mode = 'here_no_doodad'

    print(exp_name)

    search_space = {
        'seed': list(range(nseeds)),
    }
    variant = DEFAULT_PEARL_CONFIG.copy()
    variant["env_name"] = "cheetah-vel"
    variant['trainer_kwargs']["train_context_decoder"] = True
    variant["saved_tasks_path"] = "examples/smac/cheetah_tasks.joblib"  # TODO: update to point to correct file

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant['exp_id'] = exp_id
        run_experiment(
            pearl_experiment,
            unpack_variant=True,
            exp_prefix=exp_name,
            mode=mode,
            variant=variant,
            time_in_mins=3 * 24 * 60 - 1,
            use_gpu=gpu,
        )


if __name__ == "__main__":
    main()

