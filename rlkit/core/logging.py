"""
Based on rllab's logger.

https://github.com/rll/rllab
"""
import csv
import datetime
import errno
import joblib
import json
import os
import os.path as osp
import pickle
import sys
import torch
from collections import OrderedDict
from contextlib import contextmanager
from enum import Enum

import dateutil.tz
import dateutil.tz
import numpy as np
import uuid

from rlkit.core.tabulate import tabulate
from rlkit import pythonplusplus as ppp


def reopen(f):
    f.close()
    return open(f.name, 'a')


def add_prefix(log_dict: OrderedDict, prefix: str, divider=''):
    with_prefix = OrderedDict()
    for key, val in log_dict.items():
        with_prefix[prefix + divider + key] = val
    return with_prefix


def append_log(log_dict, to_add_dict, prefix=None, divider=''):
    if prefix is not None:
        to_add_dict = add_prefix(to_add_dict, prefix=prefix, divider=divider)
    return log_dict.update(to_add_dict)


class TerminalTablePrinter(object):
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Logger(object):
    def __init__(self):
        self.reopen_files_on_flush = False  # useful for Azure blobfuse
        self._prefixes = []
        self._prefix_str = ''

        self._tabular_prefixes = []
        self._tabular_prefix_str = ''

        self._tabular = []

        self._text_outputs = []
        self._tabular_outputs = []
        self._tabular_keys = None

        self._text_fds = {}
        self._tabular_fds = {}
        self._tabular_header_written = set()

        self._snapshot_dir = None
        self._snapshot_mode = 'all'
        self._snapshot_gap = 1

        self._log_tabular_only = False
        self._header_printed = False
        self.table_printer = TerminalTablePrinter()

        self._use_tensorboard = False
        self.epoch = 0

        self._save_param_mode = 'torch'

    def reset(self):
        self.__init__()

    def _add_output(self, file_name, arr, fds, mode='a'):
        if file_name not in arr:
            mkdir_p(os.path.dirname(file_name))
            arr.append(file_name)
            fds[file_name] = open(file_name, mode)

    def _remove_output(self, file_name, arr, fds):
        if file_name in arr:
            fds[file_name].close()
            del fds[file_name]
            arr.remove(file_name)
        self._tabular_keys=None

    def push_prefix(self, prefix):
        self._prefixes.append(prefix)
        self._prefix_str = ''.join(self._prefixes)

    def add_text_output(self, file_name):
        self._add_output(file_name, self._text_outputs, self._text_fds,
                         mode='a')

    def add_tensorboard_output(self, file_name):
        import tensorboard_logger
        self._use_tensorboard = True
        self.tensorboard_logger = tensorboard_logger.Logger(file_name)

    def remove_text_output(self, file_name):
        self._remove_output(file_name, self._text_outputs, self._text_fds)

    def add_tabular_output(self, file_name, relative_to_snapshot_dir=False):
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        self._add_output(file_name, self._tabular_outputs, self._tabular_fds,
                         mode='w')

    def remove_tabular_output(self, file_name, relative_to_snapshot_dir=False):
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        if self._tabular_fds[file_name] in self._tabular_header_written:
            self._tabular_header_written.remove(self._tabular_fds[file_name])
        self._remove_output(file_name, self._tabular_outputs, self._tabular_fds)

    def set_snapshot_dir(self, dir_name):
        self._snapshot_dir = dir_name

    def get_snapshot_dir(self, ):
        return self._snapshot_dir

    def get_snapshot_mode(self, ):
        return self._snapshot_mode

    def set_snapshot_mode(self, mode):
        self._snapshot_mode = mode

    def get_snapshot_gap(self, ):
        return self._snapshot_gap

    def set_snapshot_gap(self, gap):
        self._snapshot_gap = gap

    def get_save_param_mode(self, ):
        return self._save_param_mode

    def set_save_param_mode(self, mode):
        assert mode in ['pickle', 'torch', 'joblib']
        self._save_param_mode = mode

    def set_log_tabular_only(self, log_tabular_only):
        self._log_tabular_only = log_tabular_only

    def get_log_tabular_only(self, ):
        return self._log_tabular_only

    def log(self, s, with_prefix=True, with_timestamp=True):
        out = s
        if with_prefix:
            out = self._prefix_str + out
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            out = "%s | %s" % (timestamp, out)
        if not self._log_tabular_only:
            # Also log to stdout
            print(out)
            for fd in list(self._text_fds.values()):
                fd.write(out + '\n')
                fd.flush()
            if self.reopen_files_on_flush:
                self._text_fds = {
                    k: reopen(fd) for k, fd in self._text_fds.items()
                }
            sys.stdout.flush()

    def record_tabular(self, key, val):
        self._tabular.append((self._tabular_prefix_str + str(key), str(val)))
        if self._use_tensorboard:
            self.tensorboard_logger.log_value(self._tabular_prefix_str + str(key), val, self.epoch)

    def record_dict(self, d, prefix=None):
        if prefix is not None:
            self.push_tabular_prefix(prefix)
        for k, v in d.items():
            self.record_tabular(k, v)
        if prefix is not None:
            self.pop_tabular_prefix()

    def push_tabular_prefix(self, key):
        self._tabular_prefixes.append(key)
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def pop_tabular_prefix(self, ):
        del self._tabular_prefixes[-1]
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def save_extra_data(self, data, file_name='extra_data', mode='joblib'):
        """
        Data saved here will always override the last entry

        :param data: Something pickle'able.
        """
        file_name = osp.join(self._snapshot_dir, file_name)
        self._save_params_to_file(data, file_name, mode=mode)
        return file_name

    def get_table_dict(self, ):
        return dict(self._tabular)

    def get_table_key_set(self, ):
        return set(key for key, value in self._tabular)

    @contextmanager
    def prefix(self, key):
        self.push_prefix(key)
        try:
            yield
        finally:
            self.pop_prefix()

    @contextmanager
    def tabular_prefix(self, key):
        self.push_tabular_prefix(key)
        yield
        self.pop_tabular_prefix()

    def log_variant(self, log_file, variant_data):
        mkdir_p(os.path.dirname(log_file))
        with open(log_file, "w") as f:
            json.dump(variant_data, f, indent=2, sort_keys=True, cls=MyEncoder)

    def record_tabular_misc_stat(self, key, values, placement='back'):
        if placement == 'front':
            prefix = ""
            suffix = key
        else:
            prefix = key
            suffix = ""
        if len(values) > 0:
            self.record_tabular(prefix + "Average" + suffix, np.average(values))
            self.record_tabular(prefix + "Std" + suffix, np.std(values))
            self.record_tabular(prefix + "Median" + suffix, np.median(values))
            self.record_tabular(prefix + "Min" + suffix, np.min(values))
            self.record_tabular(prefix + "Max" + suffix, np.max(values))
        else:
            self.record_tabular(prefix + "Average" + suffix, np.nan)
            self.record_tabular(prefix + "Std" + suffix, np.nan)
            self.record_tabular(prefix + "Median" + suffix, np.nan)
            self.record_tabular(prefix + "Min" + suffix, np.nan)
            self.record_tabular(prefix + "Max" + suffix, np.nan)

    def dump_tabular(self, *args, **kwargs):
        self.epoch += 1
        wh = kwargs.pop("write_header", None)

        if len(self._tabular) > 0:
            if self._log_tabular_only:
                self.table_printer.print_tabular(self._tabular)
            else:
                for line in tabulate(self._tabular).split('\n'):
                    self.log(line, *args, **kwargs)
            tabular_dict = dict(self._tabular)

            # Only saves keys in first iteration to CSV!
            # (But every key is printed out in text)
            if self._tabular_keys is None:
                self._tabular_keys = list(sorted(tabular_dict.keys()))

            # Write to the csv files
            for tabular_fd in list(self._tabular_fds.values()):
                writer = csv.DictWriter(tabular_fd,
                                        fieldnames=self._tabular_keys,
                                        extrasaction="ignore",)
                if wh or (
                        wh is None and tabular_fd not in self._tabular_header_written):
                    writer.writeheader()
                    self._tabular_header_written.add(tabular_fd)
                writer.writerow(tabular_dict)
                tabular_fd.flush()
            if self.reopen_files_on_flush:
                new_tabular_fds = {}
                for k, fd in self._tabular_fds.items():
                    new_fd = reopen(fd)
                    new_tabular_fds[k] = new_fd
                    if fd in self._tabular_header_written:
                        self._tabular_header_written.remove(fd)
                        self._tabular_header_written.add(new_fd)
                self._tabular_fds = new_tabular_fds
            del self._tabular[:]

    def pop_prefix(self, ):
        del self._prefixes[-1]
        self._prefix_str = ''.join(self._prefixes)

    def _save_params_to_file(self, params, file_name, mode):
        if mode == 'joblib':
            joblib.dump(params, file_name + ".pkl", compress=3)
        elif mode == 'pickle':
            pickle.dump(params, open(file_name + ".pkl", "wb"))
        elif mode == 'cloudpickle':
            import cloudpickle
            cloudpickle.dump(params, open(file_name + ".cpkl", "wb"))
            print(file_name + ".cpkl", "wb")
        elif mode == 'torch':
            torch.save(params, file_name + ".pt")
        elif mode == 'txt':
            with open(file_name + ".txt", "w") as f:
                f.write(params)
        else:
            raise ValueError("Invalid mode: {}".format(mode))

    def save_itr_params(self, itr, params):
        if self._snapshot_dir:
            if self._snapshot_mode == 'all':
                file_name = osp.join(self._snapshot_dir, 'itr_%d' % itr)
                self._save_params_to_file(params, file_name, mode=self._save_param_mode)
            elif self._snapshot_mode == 'last':
                # override previous params
                file_name = osp.join(self._snapshot_dir, 'params')
                self._save_params_to_file(params, file_name, mode=self._save_param_mode)
            elif self._snapshot_mode == "gap":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir, 'itr_%d' % itr)
                    self._save_params_to_file(params, file_name, mode=self._save_param_mode)
            elif self._snapshot_mode == "gap_and_last":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir, 'itr_%d' % itr)
                    self._save_params_to_file(params, file_name, mode=self._save_param_mode)
                file_name = osp.join(self._snapshot_dir, 'params')
                self._save_params_to_file(params, file_name, mode=self._save_param_mode)
            elif self._snapshot_mode == 'none':
                pass
            else:
                raise NotImplementedError


def setup_logger(
        logger,
        exp_name,
        base_log_dir,
        variant=None,
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        snapshot_gap=1,
        log_tabular_only=False,
        log_dir=None,
        tensorboard=False,
        unique_id=None,
        git_infos=None,
        script_name=None,
        run_id=None,
        first_time=True,
        reopen_files_on_flush=False,
        **create_log_dir_kwargs
):
    """
    Set up logger to have some reasonable default settings.
    Will save log output to
        based_log_dir/exp_name/exp_name.
    exp_name will be auto-generated to be unique.
    If log_dir is specified, then that directory is used as the output dir.
    :param exp_name: The sub-directory for this specific experiment.
    :param variant:
    :param base_log_dir: The directory where all log should be saved.
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :param log_dir:
    :return:
    """
    logger.reset()
    variant = variant or {}
    unique_id = unique_id or str(uuid.uuid4())

    if log_dir is None:
        log_dir = create_log_dir(
            exp_name=exp_name,
            base_log_dir=base_log_dir,
            variant=variant,
            run_id=run_id,
            **create_log_dir_kwargs
        )

    if tensorboard:
        tensorboard_log_path = osp.join(log_dir, "tensorboard")
        logger.add_tensorboard_output(tensorboard_log_path)

    logger.log("Variant:")
    variant_to_save = variant.copy()
    variant_to_save['unique_id'] = unique_id
    variant_to_save['exp_name'] = exp_name
    variant_to_save['trial_name'] = log_dir.split('/')[-1]
    logger.log(
        json.dumps(ppp.dict_to_safe_json(variant_to_save, sort=True), indent=2)
    )
    variant_log_path = osp.join(log_dir, variant_log_file)
    logger.log_variant(variant_log_path, variant_to_save)

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
    logger.reopen_files_on_flush = reopen_files_on_flush
    exp_name = log_dir.split("/")[-1]
    logger.push_prefix("[%s] " % exp_name)

    if git_infos:
        save_git_infos(git_infos, log_dir)
    if script_name:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)
    return log_dir


def save_git_infos(git_infos, log_dir):
    for (
            directory, code_diff, code_diff_staged, commit_hash, branch_name
    ) in git_infos:
        if directory[-1] == '/':
            diff_file_name = directory[1:-1].replace("/", "-") + ".patch"
            diff_staged_file_name = (
                    directory[1:-1].replace("/", "-") + "_staged.patch"
            )
        else:
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
            f.write("directory: {}".format(directory))
            f.write('\n')
            f.write("git hash: {}".format(commit_hash))
            f.write('\n')
            f.write("git branch name: {}".format(branch_name))
            f.write('\n\n')


def create_log_dir(
        exp_name,
        base_log_dir,
        exp_id=0,
        seed=0,
        variant=None,
        trial_dir_suffix=None,
        add_time_suffix=True,
        include_exp_name_sub_dir=True,
        run_id=None,
):
    """
    Creates and returns a unique log directory.
    :param exp_name: All experiments with this prefix will have log
    directories be under this directory.
    :param exp_id: Different exp_ids will be in different directories.
    :return:
    """
    if run_id is not None:
        exp_id = variant["exp_id"]
        if variant.get("num_exps_per_instance", 0) > 1:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
            trial_name = "run%s/id%s/%s--s%d" % (run_id, exp_id, timestamp, seed)
        else:
            trial_name = "run{}/id{}".format(run_id, exp_id)
    else:
        trial_name = create_trial_name(exp_name, exp_id=exp_id, seed=seed, add_time_suffix=add_time_suffix)
    if trial_dir_suffix is not None:
        trial_name = "{}-{}".format(trial_name, trial_dir_suffix)
    if include_exp_name_sub_dir:
        log_dir = osp.join(base_log_dir, exp_name.replace("_", "-"), trial_name)
    else:
        log_dir = osp.join(base_log_dir, trial_name)
    if osp.exists(log_dir):
        print("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def create_trial_name(exp_name, exp_id=0, seed=0, add_time_suffix=True):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_name:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    if add_time_suffix:
        return "%s_%s_id%03d--s%d" % (exp_name, timestamp, exp_id, seed)
    else:
        return "%s_id%03d--s%d" % (exp_name, exp_id, seed)


logger = Logger()
