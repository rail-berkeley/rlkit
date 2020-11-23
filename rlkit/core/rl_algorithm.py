import abc
from collections import OrderedDict

import gtimer as gt
import wandb

from rlkit.core import eval_util, logger
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times["time/{} (s)".format(key)] = time
    times["time/epoch (s)"] = epoch_time
    times["time/total (s)"] = gt.get_times().total
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
        use_wandb: bool = True,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0

        self.post_epoch_funcs = []
        self.use_wandb = use_wandb
        if use_wandb:
            for network in trainer.networks:
                wandb.watch(network, log="all", log_freq=1)

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError("_train must implemented by inherited class")

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp("saving")
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot["trainer/" + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot["exploration/" + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot["evaluation/" + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot["replay_buffer/" + k] = v
        return snapshot

    def _log_wandb(self, d, prefix, epoch):
        updated_d = {}
        for k, v in d.items():
            updated_d[prefix + k] = v
        wandb.log(updated_d, step=epoch)

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(), prefix="replay_buffer/"
        )
        if self.use_wandb:
            self._log_wandb(
                self.replay_buffer.get_diagnostics(),
                prefix="replay_buffer/",
                epoch=epoch,
            )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix="trainer/")
        if self.use_wandb:
            self._log_wandb(
                self.trainer.get_diagnostics(), prefix="trainer/", epoch=epoch
            )

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(), prefix="exploration/"
        )
        if self.use_wandb:
            self._log_wandb(
                self.expl_data_collector.get_diagnostics(),
                prefix="exploration/",
                epoch=epoch,
            )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, "get_diagnostics"):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths), prefix="exploration/",
            )
            if self.use_wandb:
                self._log_wandb(
                    self.expl_env.get_diagnostics(expl_paths),
                    prefix="exploration/",
                    epoch=epoch,
                )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths), prefix="exploration/",
        )
        if self.use_wandb:
            self._log_wandb(
                eval_util.get_generic_path_information(expl_paths),
                prefix="exploration/",
                epoch=epoch,
            )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(), prefix="evaluation/",
        )
        if self.use_wandb:
            self._log_wandb(
                self.eval_data_collector.get_diagnostics(),
                prefix="evaluation/",
                epoch=epoch,
            )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, "get_diagnostics"):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths), prefix="evaluation/",
            )
            if self.use_wandb:
                self._log_wandb(
                    self.eval_env.get_diagnostics(eval_paths),
                    prefix="evaluation/",
                    epoch=epoch,
                )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths), prefix="evaluation/",
        )
        if self.use_wandb:
            self._log_wandb(
                eval_util.get_generic_path_information(eval_paths),
                prefix="evaluation/",
                epoch=epoch,
            )

        """
        Misc
        """
        gt.stamp("logging")
        logger.record_dict(_get_epoch_timings())
        if self.use_wandb:
            self._log_wandb(
                _get_epoch_timings(), prefix="", epoch=epoch,
            )
        logger.record_tabular("Epoch", epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
