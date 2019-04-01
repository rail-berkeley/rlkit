import numpy as np

from rlkit.torch.torch_rl_algorithm import TorchTrainer


class HERTrainer(TorchTrainer):
    def __init__(self, base_trainer: TorchTrainer):
        self._base_trainer = base_trainer

    def train(self, data):
        obs = data['observations']
        next_obs = data['next_observations']
        goals = data['resampled_goals']
        data['observations'] = np.hstack((
            obs,
            goals
        ))
        data['next_observations'] = np.hstack((
            next_obs,
            goals,
        ))
        self._base_trainer.train(data)

    def get_diagnostics(self):
        return self._base_trainer.get_diagnostics()

    def end_epoch(self, epoch):
        self._base_trainer.end_epoch(epoch)

    @property
    def networks(self):
        return self._base_trainer.networks

    def get_snapshot(self):
        return self._base_trainer.get_snapshot()

