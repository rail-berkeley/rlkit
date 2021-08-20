import datetime
import json
import os
import os.path as osp
import pickle
import sys
import time
from typing import NamedTuple
import random
from collections import namedtuple

import __main__ as main
import dateutil.tz
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger, setup_logger_custom
from rlkit.launchers import conf as config
import torch
from rlkit import pythonplusplus as ppp


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
        unpack_variant=False,
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
    setup_logger_custom(
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

def create_exp_name(exp_prefix, exp_id=0, seed=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return "%s_%s_%04d--s-%d" % (exp_prefix, timestamp, exp_id, seed)


def create_log_dir(
        exp_prefix,
        exp_id=0,
        seed=0,
        base_log_dir=None,
        include_exp_prefix_sub_dir=True,
):
    """
    Creates and returns a unique log directory.
    :param exp_prefix: All experiments with this prefix will have log
    directories be under this directory.
    :param exp_id: The number of the specific experiment run within this
    experiment.
    :param base_log_dir: The directory where all log should be saved.
    :return:
    """
    exp_name = create_exp_name(exp_prefix, exp_id=exp_id,
                               seed=seed)
    if base_log_dir is None:
        base_log_dir = conf.LOCAL_LOG_DIR
    if include_exp_prefix_sub_dir:
        log_dir = osp.join(base_log_dir, exp_prefix.replace("_", "-"), exp_name)
    else:
        log_dir = osp.join(base_log_dir, exp_name)
    if osp.exists(log_dir):
        print("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setup_logger(
        exp_prefix="default",
        variant=None,
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        snapshot_gap=1,
        log_tabular_only=False,
        log_dir=None,
        git_infos=None,
        script_name=None,
        **create_log_dir_kwargs
):
    """
    Set up logger to have some reasonable default settings.
    Will save log output to
        based_log_dir/exp_prefix/exp_name.
    exp_name will be auto-generated to be unique.
    If log_dir is specified, then that directory is used as the output dir.
    :param exp_prefix: The sub-directory for this specific experiment.
    :param variant:
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :param log_dir:
    :param git_infos:
    :param script_name: If set, save the script name to this.
    :return:
    """
    if git_infos is None:
        git_infos = get_git_infos(conf.CODE_DIRS_TO_MOUNT)
    first_time = log_dir is None
    if first_time:
        log_dir = create_log_dir(exp_prefix, **create_log_dir_kwargs)

    if variant is not None:
        logger.log("Variant:")
        logger.log(json.dumps(ppp.dict_to_safe_json(variant), indent=2))
        variant_log_path = osp.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    logger.add_text_output(text_log_path)
    if first_time:
        logger.add_tabular_output(tabular_log_path)
    else:
        logger._add_output(tabular_log_path, logger._tabular_outputs,
                           logger._tabular_fds, mode='a')
        for tabular_fd in logger._tabular_fds:
            logger._tabular_header_written.add(tabular_fd)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    exp_name = log_dir.split("/")[-1]
    logger.push_prefix("[%s] " % exp_name)

    if git_infos is not None:
        for (
            directory, code_diff, code_diff_staged, commit_hash, branch_name
        ) in git_infos:
            if directory[-1] == '/':
                directory = directory[:-1]
            diff_file_name = directory[1:].replace("/", "-") + ".patch"
            diff_staged_file_name = (
                directory[1:].replace("/", "-") + "_staged.patch"
            )
            if code_diff is not None and len(code_diff) > 0:
                with open(osp.join(log_dir, diff_file_name), "w") as f:
                    f.write(code_diff + '\n')
            if code_diff_staged is not None and len(code_diff_staged) > 0:
                with open(osp.join(log_dir, diff_staged_file_name), "w") as f:
                    f.write(code_diff_staged + '\n')
            with open(osp.join(log_dir, "git_infos.txt"), "a") as f:
                f.write("directory: {}\n".format(directory))
                f.write("git hash: {}\n".format(commit_hash))
                f.write("git branch name: {}\n\n".format(branch_name))
    if script_name is not None:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)
    return log_dir


GitInfo = namedtuple(
    'GitInfo',
    [
        'directory',
        'code_diff',
        'code_diff_staged',
        'commit_hash',
        'branch_name',
    ],
)


def get_git_infos(dirs):
    try:
        import git
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
            except git.exc.InvalidGitRepositoryError as e:
                print("Not a valid git repo: {}".format(directory))
    except ImportError:
        git_infos = None
    return git_infos
