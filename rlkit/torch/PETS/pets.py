from collections import OrderedDict

import torch.optim as optim
import torch.nn as nn

from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.core import np_ify
from rlkit.torch.PETS.model import gaussian_log_loss


class PETSTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            model,
            lr=1e-3,
            optimizer_class=optim.Adam,

            # discount=0.99,
            reward_scale=1.0,
            plotter=None,
            render_eval_paths=False
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.model = model
        self.lr = lr

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.reward_criterion = nn.MSELoss() if model.predict_reward else None
        self.mean_criterion = nn.MSELoss()  # just for information, not for training
        self.model_criterion = gaussian_log_loss
        self.model_optimizer = optimizer_class(
                self.model.parameters(),
                lr=lr
        )
        # self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        # terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # in order to bootstrap the models, we need to train one network only per batch
        net_idx = self._n_train_steps_total % len(self.model._nets)
        mean, logvar, predcted_rewards = self.model.forward(obs, actions, network_idx=net_idx, return_net_outputs=True)
        # TODO: possibly need to include weight decay
        mean_mse = self.mean_criterion(mean, next_obs)

        model_loss = self.model_criterion(mean, logvar, next_obs)
        bound_loss = self.model.bound_loss()
        if self.reward_criterion:
            reward_loss = self.reward_criterion(predcted_rewards, rewards)
        else:
            reward_loss = 0
        loss = model_loss + bound_loss + reward_loss
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        self.model.trained_at_all = True

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['Model Loss'] = np_ify(model_loss)
            self.eval_statistics['Bound Loss'] = np_ify(bound_loss)
            self.eval_statistics['Reward Loss'] = np_ify(reward_loss)
            self.eval_statistics['Model MSE'] = np_ify(mean_mse)
            self.eval_statistics['Loss'] = np_ify(loss)
        self._n_train_steps_total += 1

    @property
    def networks(self):
        return [self.model]
        # return self.model._nets

    def get_snapshot(self):
        return dict(model=self.model)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
