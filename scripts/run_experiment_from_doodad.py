import doodad as dd
from rlkit.launchers.launcher_util import run_experiment_here
import torch.multiprocessing as mp

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('agg')

    mp.set_start_method('forkserver')
    args_dict = dd.get_args()
    method_call = args_dict['method_call']
    run_experiment_kwargs = args_dict['run_experiment_kwargs']
    output_dir = args_dict['output_dir']
    run_mode = args_dict.get('mode', None)
    if run_mode and run_mode in ['slurm_singularity', 'sss']:
        import os
        run_experiment_kwargs['variant']['slurm-job-id'] = os.environ.get(
            'SLURM_JOB_ID', None
        )
    if run_mode and run_mode == 'ec2':
        try:
            import urllib.request
            instance_id = urllib.request.urlopen(
                'http://169.254.169.254/latest/meta-data/instance-id'
            ).read().decode()
            run_experiment_kwargs['variant']['EC2_instance_id'] = instance_id
        except Exception as e:
            print("Could not get instance ID. Error was...")
            print(e)
    if run_mode and (run_mode == 'ec2' or run_mode == 'gcp'):
        # Do this in case base_log_dir was already set
        run_experiment_kwargs['base_log_dir'] = output_dir
        run_experiment_here(
            method_call,
            include_exp_prefix_sub_dir=False,
            **run_experiment_kwargs
        )
    else:
        run_experiment_here(
            method_call,
            log_dir=output_dir,
            **run_experiment_kwargs
        )