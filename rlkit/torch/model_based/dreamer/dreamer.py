from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from rlkit.core.loss import LossFunction, LossStatistics
from torch import nn as nn
import torch.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix
import gtimer as gt

DreamerLosses = namedtuple(
    'DreamerLosses',
    'actor_loss vf_loss world_model_loss',
)

class DreamerTrainer(TorchTrainer, LossFunction):
    def __init__(
            self,
            env,
            actor,
            vf,
            world_model,

            discount=0.99,
            reward_scale=1.0,

            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=6e-4,

            optimizer_class=optim.Adam,

            gradient_clip=100.0,
            lam=.95,
            imagination_horizon=2,
            free_nats=3.0,
            kl_scale=1.0,

            plotter=None,
            render_eval_paths=False,
    ):
        super().__init__()
        self.env = env
        self.actor = actor
        self.world_model = world_model
        self.vf = vf

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.actor_optimizer = optimizer_class(
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )

        self.world_model_optimizer = optimizer_class(
            self.world_model.parameters(),
            lr=world_model_lr,
        )
        self.discount = discount
        self.reward_scale = reward_scale
        self.gradient_clip=gradient_clip
        self.lam=lam
        self.imagination_horizon=imagination_horizon
        self.free_nats=free_nats
        self.kl_scale=kl_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train_from_torch(self, batch):
        gt.blank_stamp()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """

        self.actor_optimizer.zero_grad()
        losses.actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip, norm_type=2)
        self.actor_optimizer.step()

        self.vf_optimizer.zero_grad()
        losses.vf_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.vf.parameters(), self.gradient_clip, norm_type=2)
        self.vf_optimizer.step()

        self.world_model_optimizer.zero_grad()
        losses.world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.gradient_clip, norm_type=2)
        self.world_model_optimizer.step()

        self._n_train_steps_total += 1

        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp('dreamer training', unique=False)
    
    def imagine_ahead(self, state):
        for k,v in state.items():
            state[k] = state[k].reshape(-1, state[k].shape[-1])
        feat_full = []
        for i in range(self.imagination_horizon):
            feat = self.world_model.get_feat(state).detach()
            action = self.actor(feat).sample()
            state = self.world_model.img_step(state, action)
            feat_full.append(self.world_model.get_feat(state).unsqueeze(0))
        return torch.cat(feat_full)

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
        **kwargs,
    ) -> Tuple[DreamerLosses, LossStatistics]:
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        World Model Loss
        """
        post, prior, post_dist, prior_dist, image_dist, reward_dist = self.world_model(obs, actions, loop_through_path_length=True)
        image_pred_loss = image_dist.log_prob(self.world_model.preprocess(obs).reshape(-1, 3, 64, 64)).mean()
        reward_pred_loss = reward_dist.log_prob(rewards.reshape(-1, 1)).mean()
        div = torch.distributions.kl_divergence(post_dist, prior_dist).mean()
        div = div.clamp_max(self.free_nats)
        model_loss = self.kl_scale * div - (image_pred_loss + reward_pred_loss)

        """
        Policy Loss
        """
        imag_feat = self.imagine_ahead(post)
        imagined_reward = self.world_model.reward(imag_feat)
        pcont = self.discount * torch.ones_like(imagined_reward)
        value = self.vf(imag_feat)
        imagined_returns = lambda_return(
            imagined_reward[:-1], value[:-1],
            bootstrap=value[-1], lambda_=self.lam,)
        discount = torch.cumprod(torch.cat(
            [torch.ones_like(pcont[:1]), pcont[:-2]], 0), 0).detach()
        actor_loss = -(discount * imagined_returns).mean()

        """
        Value Loss
        """
        value_pred = self.world_model.get_dist(self.vf(imag_feat)[:-1], 1)
        target = imagined_returns.detach()
        value_loss = -(discount * value_pred.log_prob(target)).mean()

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['Value Loss'] = np.mean(ptu.get_numpy(value_loss))
            eval_statistics['Actor Loss'] = np.mean(ptu.get_numpy(
                actor_loss
            ))
            eval_statistics['World Model Loss'] = np.mean(ptu.get_numpy(
                model_loss
            ))
            eval_statistics['Image Loss'] = -1*np.mean(ptu.get_numpy(image_pred_loss))
            eval_statistics['Reward Loss'] = -1*np.mean(ptu.get_numpy(reward_pred_loss))
            eval_statistics['Divergence Loss'] = np.mean(ptu.get_numpy(div))
            eval_statistics['Imagined Returns'] = np.mean(ptu.get_numpy(imagined_returns.mean()))

        loss = DreamerLosses(
            actor_loss=actor_loss,
            world_model_loss=model_loss,
            vf_loss=value_loss,
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.actor,
            self.vf,
            self.world_model,
        ]

    @property
    def optimizers(self):
        return [
            self.actor_optimizer,
            self.vf_optimizer,
            self.world_model_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            actor=self.actor,
            world_model=self.world_model,
            vf=self.vf,
        )

def lambda_return(imged_reward, value_pred, bootstrap, discount=0.99, lambda_=0.95):
    #from: https://github.com/yusukeurakami/dreamer-pytorch
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
    discount_tensor = discount * torch.ones_like(imged_reward) #pcont
    inputs = imged_reward + discount_tensor * next_values * (1 - lambda_)
    last = bootstrap
    indices = reversed(range(len(inputs)))
    outputs = []
    for index in indices:
        inp, disc = inputs[index], discount_tensor[index]
        last = inp + disc*lambda_*last
        outputs.append(last)
    outputs = list(reversed(outputs))
    outputs = torch.stack(outputs, 0)
    returns = outputs
    return returns