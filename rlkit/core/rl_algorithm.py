import abc
from collections import OrderedDict

from rlkit.core.timer import timer

from rlkit.core import logger, eval_util
from rlkit.core.logging import append_log
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector


def _get_epoch_timings():
    times_itrs = timer.get_times()
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: DataCollector,
            evaluation_data_collector: DataCollector,
            replay_buffer: ReplayBuffer,
            num_epochs,
            exploration_get_diagnostic_functions=None,
            evaluation_get_diagnostic_functions=None,
            eval_epoch_freq=1,
            eval_only=False,
            save_algorithm=False,
            save_replay_buffer=False,
            save_logger=False,
            save_extra_manual_epoch_list=(),
            keep_only_last_extra=True,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0
        self.post_train_funcs = []
        self.post_epoch_funcs = []
        self.epoch = self._start_epoch
        self.num_epochs = num_epochs
        self.save_algorithm = save_algorithm
        self.save_replay_buffer = save_replay_buffer
        self.save_extra_manual_epoch_list = save_extra_manual_epoch_list
        self.save_logger = save_logger
        self.keep_only_last_extra = keep_only_last_extra
        if exploration_get_diagnostic_functions is None:
            exploration_get_diagnostic_functions = [
                eval_util.get_generic_path_information,
            ]
            if hasattr(self.expl_env, 'get_diagnostics'):
                exploration_get_diagnostic_functions.append(
                    self.expl_env.get_diagnostics)
        if evaluation_get_diagnostic_functions is None:
            evaluation_get_diagnostic_functions = [
                eval_util.get_generic_path_information,
            ]
            if hasattr(self.eval_env, 'get_diagnostics'):
                evaluation_get_diagnostic_functions.append(
                    self.eval_env.get_diagnostics)
        self._eval_get_diag_fns = evaluation_get_diagnostic_functions
        self._expl_get_diag_fns = exploration_get_diagnostic_functions

        self._eval_epoch_freq = eval_epoch_freq
        self._eval_only = eval_only

    def train(self):
        timer.return_global_times = True
        for _ in range(self.epoch, self.num_epochs):
            self._begin_epoch()
            timer.start_timer('saving')
            logger.save_itr_params(self.epoch, self._get_snapshot())
            timer.stop_timer('saving')
            log_dict, _ = self._train()
            logger.record_dict(log_dict)
            logger.dump_tabular(with_prefix=True, with_timestamp=False)
            self._end_epoch()
        logger.save_itr_params(self.epoch, self._get_snapshot())

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _begin_epoch(self):
        timer.reset()

    def _end_epoch(self):
        for post_train_func in self.post_train_funcs:
            post_train_func(self, self.epoch)

        self.expl_data_collector.end_epoch(self.epoch)
        self.eval_data_collector.end_epoch(self.epoch)
        self.replay_buffer.end_epoch(self.epoch)
        self.trainer.end_epoch(self.epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, self.epoch)

        if self.epoch in self.save_extra_manual_epoch_list:
            if self.keep_only_last_extra:
                file_name = 'extra_snapshot'
                info_lines = [
                    'extra_snapshot_itr = {}'.format(self.epoch),
                    'snapshot_dir = {}'.format(logger.get_snapshot_dir())
                ]
                logger.save_extra_data(
                    '\n'.join(info_lines),
                    file_name='snapshot_info',
                    mode='txt',
                )
            else:
                file_name = 'extra_snapshot_itr{}'.format(self.epoch)
            logger.save_extra_data(
                self.get_extra_data_to_save(self.epoch),
                file_name=file_name,
                mode='cloudpickle',
            )
        self.epoch += 1

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        if self.save_logger:
            data_to_save['logger'] = logger
        return data_to_save

    def _get_diagnostics(self):
        timer.start_timer('logging', unique=False)
        algo_log = OrderedDict()
        append_log(algo_log, self.replay_buffer.get_diagnostics(),
                   prefix='replay_buffer/')
        append_log(algo_log, self.trainer.get_diagnostics(), prefix='trainer/')
        # Exploration
        append_log(algo_log, self.expl_data_collector.get_diagnostics(),
                   prefix='expl/')
        expl_paths = self.expl_data_collector.get_epoch_paths()
        for fn in self._expl_get_diag_fns:
            append_log(algo_log, fn(expl_paths), prefix='expl/')
        # Eval
        if self.epoch % self._eval_epoch_freq == 0:
            self._prev_eval_log = OrderedDict()
            eval_diag = self.eval_data_collector.get_diagnostics()
            self._prev_eval_log.update(eval_diag)
            append_log(algo_log, eval_diag, prefix='eval/')
            eval_paths = self.eval_data_collector.get_epoch_paths()
            for fn in self._eval_get_diag_fns:
                addl_diag = fn(eval_paths)
                self._prev_eval_log.update(addl_diag)
                append_log(algo_log, addl_diag, prefix='eval/')
        else:
            append_log(algo_log, self._prev_eval_log, prefix='eval/')

        append_log(algo_log, _get_epoch_timings())
        algo_log['epoch'] = self.epoch
        try:
            import os
            import psutil
            process = psutil.Process(os.getpid())
            algo_log['RAM Usage (Mb)'] = int(process.memory_info().rss / 1000000)
        except ImportError:
            pass
        timer.stop_timer('logging')
        return algo_log

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
