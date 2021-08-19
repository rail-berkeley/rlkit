import os
import time
from typing import NamedTuple
import random

import __main__ as main
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger, setup_logger
from rlkit.launchers import config
import torch


GitInfo = NamedTuple(
    'GitInfo',
    [
        ('directory', str),
        ('code_diff', str),
        ('code_diff_staged', str),
        ('commit_hash', str),
        ('branch_name', str),
    ],
)

class AutoSetup:
    """
    Automatically set up:
    1. the logger
    2. the GPU mode
    3. the seed
    :param exp_function: some function that should not depend on `logger_config`
    nor `seed`.
    :param unpack_variant: do you call exp_function with `**variant`?
    :return: function output
    """
    def __init__(self, exp_function, unpack_variant=True):
        self.exp_function = exp_function
        self.unpack_variant = unpack_variant

    def __call__(self, doodad_config, variant):
        if doodad_config:
            variant_to_save = variant.copy()
            variant_to_save['doodad_info'] = doodad_config.extra_launch_info
            setup_experiment(
                variant=variant_to_save,
                exp_name=doodad_config.exp_name,
                base_log_dir=doodad_config.base_log_dir,
                git_infos=doodad_config.git_infos,
                script_name=doodad_config.script_name,
                use_gpu=doodad_config.use_gpu,
                gpu_id=doodad_config.gpu_id,
            )
        variant.pop('logger_config', None)
        variant.pop('seed', None)
        variant.pop('exp_id', None)
        variant.pop('run_id', None)
        if self.unpack_variant:
            self.exp_function(**variant)
        else:
            self.exp_function(variant)

def run_experiment(
        method_call,
        exp_name='default',
        mode='local',
        variant=None,
        use_gpu=False,
        gpu_id=0,
        wrap_fn_with_auto_setup=True,
        unpack_variant=True,
        base_log_dir=None,
        prepend_date_to_exp_name=True,
        **kwargs
):
    if base_log_dir is None:
        base_log_dir=config.LOCAL_LOG_DIR
    if wrap_fn_with_auto_setup:
        method_call = AutoSetup(method_call, unpack_variant=unpack_variant)
    if mode == 'here_no_doodad':
        if prepend_date_to_exp_name:
            exp_name = time.strftime("%y-%m-%d") + "-" + exp_name
        setup_experiment(
            variant=variant,
            exp_name=exp_name,
            base_log_dir=base_log_dir,
            git_infos=generate_git_infos(),
            script_name=main.__file__,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
        )
        method_call(None, variant)
    else:
        from doodad.easy_launch.python_function import (
            run_experiment as doodad_run_experiment
        )
        doodad_run_experiment(
            method_call,
            exp_name=exp_name,
            mode=mode,
            variant=variant,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
            prepend_date_to_exp_name=prepend_date_to_exp_name,
            **kwargs
        )

def setup_experiment(
        variant,
        exp_name,
        base_log_dir,
        git_infos,
        script_name,
        use_gpu,
        gpu_id,
):
    logger_config = variant.get('logger_config', {})
    seed = variant.get('seed', random.randint(0, 999999))
    exp_id = variant.get('exp_id', random.randint(0, 999999))
    set_seed(seed)
    ptu.set_gpu_mode(use_gpu, gpu_id)
    os.environ['gpu_id'] = str(gpu_id)
    setup_logger(
        logger,
        exp_name=exp_name,
        base_log_dir=base_log_dir,
        variant=variant,
        git_infos=git_infos,
        script_name=script_name,
        seed=seed,
        exp_id=exp_id,
        **logger_config)


def set_seed(seed):
    """
    Set the seed for all the possible random number generators.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_git_infos():
    try:
        import git
        dirs = config.CODE_DIRS_TO_MOUNT

        git_infos = []
        for directory in dirs:
            # Idk how to query these things, so I'm just doing try-catch
            try:
                repo = git.Repo(directory)
                try:
                    branch_name = repo.active_branch.name
                except TypeError:
                    branch_name = '[DETACHED]'
                git_infos.append(GitInfo(
                    directory=directory,
                    code_diff=repo.git.diff(None),
                    code_diff_staged=repo.git.diff('--staged'),
                    commit_hash=repo.head.commit.hexsha,
                    branch_name=branch_name,
                ))
            except git.exc.InvalidGitRepositoryError:
                pass
    except (ImportError, UnboundLocalError, NameError):
        git_infos = None
    return git_infos
