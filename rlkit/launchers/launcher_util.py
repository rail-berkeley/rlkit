import datetime
import json
import os
import os.path as osp
import pickle
import random
import sys
from collections import namedtuple

import dateutil.tz
import numpy as np

from rlkit.core import logger
from rlkit.launchers import config
from rlkit.torch.pytorch_util import set_gpu_mode

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


def recursive_items(dictionary):
    """
    Get all (key, item) recursively in a potentially recursive dictionary.
    Usage:

    ```
    x = {
        'foo' : {
            'bar' : 5
        }
    }
    recursive_items(x)
    # output:
    # ('foo', {'bar' : 5})
    # ('bar', 5)
    ```
    :param dictionary:
    :return:
    """
    for key, value in dictionary.items():
        yield key, value
        if type(value) is dict:
            yield from recursive_items(value)


def save_experiment_data(dictionary, log_dir):
    with open(log_dir + '/experiment.pkl', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_experiment_here(
        experiment_function,
        variant=None,
        exp_id=0,
        seed=0,
        use_gpu=True,
        # Logger params:
        exp_prefix="default",
        snapshot_mode='last',
        snapshot_gap=1,
        git_infos=None,
        script_name=None,
        base_log_dir=None,
        log_dir=None,
):
    """
    Run an experiment locally without any serialization.

    :param experiment_function: Function. `variant` will be passed in as its
    only argument.
    :param exp_prefix: Experiment prefix for the save file.
    :param variant: Dictionary passed in to `experiment_function`.
    :param exp_id: Experiment ID. Should be unique across all
    experiments. Note that one experiment may correspond to multiple seeds,.
    :param seed: Seed used for this experiment.
    :param use_gpu: Run with GPU. By default False.
    :param script_name: Name of the running script
    :param log_dir: If set, set the log directory to this. Otherwise,
    the directory will be auto-generated based on the exp_prefix.
    :return:
    """
    if variant is None:
        variant = {}
    variant['exp_id'] = str(exp_id)

    if seed is None and 'seed' not in variant:
        seed = random.randint(0, 100000)
        variant['seed'] = str(seed)
    reset_execution_environment()

    actual_log_dir = setup_logger(
        exp_prefix=exp_prefix,
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        base_log_dir=base_log_dir,
        log_dir=log_dir,
        git_infos=git_infos,
        script_name=script_name,
    )

    set_seed(seed)
    set_gpu_mode(use_gpu)

    run_experiment_here_kwargs = dict(
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        use_gpu=use_gpu,
        exp_prefix=exp_prefix,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        git_infos=git_infos,
        script_name=script_name,
        base_log_dir=base_log_dir,
    )
    save_experiment_data(
        dict(
            run_experiment_here_kwargs=run_experiment_here_kwargs
        ),
        actual_log_dir
    )
    return experiment_function(variant)


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


def create_log_dir(exp_prefix, exp_id=0, seed=0, base_log_dir=None):
    """
    Creates and returns a unique log directory.

    :param exp_prefix: All experiments with this prefix will have log
    directories be under this directory.
    :param exp_id: Different exp_ids will be in different directories.
    :return:
    """
    exp_name = create_exp_name(exp_prefix, exp_id=exp_id,
                               seed=seed)
    if base_log_dir is None:
        base_log_dir = config.LOCAL_LOG_DIR
    log_dir = osp.join(base_log_dir, exp_prefix.replace("_", "-"), exp_name)
    if osp.exists(log_dir):
        print("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setup_logger(
        exp_prefix="default",
        exp_id=0,
        seed=0,
        variant=None,
        base_log_dir=None,
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        snapshot_gap=1,
        log_tabular_only=False,
        log_dir=None,
        git_infos=None,
        script_name=None,
):
    """
    Set up logger to have some reasonable default settings.

    Will save log output to

        based_log_dir/exp_prefix/exp_name.

    exp_name will be auto-generated to be unique.

    If log_dir is specified, then that directory is used as the output dir.

    :param exp_prefix: The sub-directory for this specific experiment.
    :param exp_id: The number of the specific experiment run within this
    experiment.
    :param variant:
    :param base_log_dir: The directory where all log should be saved.
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
        git_infos = get_git_infos(config.CODE_DIRS_TO_MOUNT)
    first_time = log_dir is None
    if first_time:
        log_dir = create_log_dir(exp_prefix, exp_id=exp_id, seed=seed,
                                 base_log_dir=base_log_dir)

    if variant is not None:
        logger.log("Variant:")
        logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
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


def dict_to_safe_json(d):
    """
    Convert each value in the dictionary into a JSON'able primitive.
    :param d:
    :return:
    """
    new_d = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False


def set_seed(seed):
    """
    Set the seed for all the possible random number generators.

    :param seed:
    :return: None
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


def reset_execution_environment():
    """
    Call this between calls to separate experiments.
    :return:
    """
    import importlib
    importlib.reload(logger)


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
