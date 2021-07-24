from rlkit.torch.smac.base_config import DEFAULT_CONFIG
from rlkit.torch.smac.launcher import smac_experiment
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

    variant = DEFAULT_CONFIG.copy()
    variant["env_name"] = "ant-dir"
    variant["env_params"]["direction_in_degrees"] = True
    search_space = {
        'load_buffer_kwargs.pretrain_buffer_path': [
            "results/.../extra_snapshot_itr100.cpkl"  # TODO: update to point to correct file
        ],
        'saved_tasks_path': [
            "examples/smac/ant_tasks.joblib",  # TODO: update to point to correct file
        ],
        'load_buffer_kwargs.start_idx': [
            -1200,
        ],
        'seed': list(range(nseeds)),
    }
    from rlkit.launchers.launcher_util import run_experiment
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

