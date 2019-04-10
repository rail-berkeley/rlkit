import abc
from typing import Iterable
from torch import nn as nn

from rlkit.core.batch_rl_algorithm import BatchRLAlgorithm
from rlkit.core.online_rl_algorithm import OnlineRLAlgorithm
from rlkit.core.trainer import Trainer


class TorchBatchRLAlgorithm(BatchRLAlgorithm, metaclass=abc.ABCMeta):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)

class TorchOnlineRLAlgorithm(OnlineRLAlgorithm, metaclass=abc.ABCMeta):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass

