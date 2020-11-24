from collections import OrderedDict, namedtuple
from typing import Tuple

import gtimer as gt
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import kl_divergence as kld

import rlkit.torch.pytorch_util as ptu
from rlkit.core.loss import LossFunction, LossStatistics
from rlkit.torch.model_based.dreamer.utils import (
    FreezeParameters,
    lambda_return,
    schedule,
    zero_grad,
)
from rlkit.torch.torch_rl_algorithm import TorchTrainer

try:
    import apex
    from apex import amp

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

DreamerLosses = namedtuple(
    "DreamerLosses",
    "actor_loss vf_loss world_model_loss",
)


class DreamerV2Trainer(TorchTrainer, LossFunction):
    def __init__(
        self,
        env,
        actor,
        vf,
        world_model,
        target_vf=None,
        discount=0.99,
        reward_scale=1.0,
        actor_lr=8e-5,
        vf_lr=8e-5,
        world_model_lr=6e-4,
        optimizer_class="torch_adam",
        use_amp=False,
        opt_level="O1",
        gradient_clip=100.0,
        lam=0.95,
        imagination_horizon=4,
        free_nats=3.0,
        kl_loss_scale=1.0,
        pred_discount_loss_scale=10.0,
        image_loss_scale=1.0,
        reward_loss_scale=1.0,
        transition_loss_scale=0.0,
        entropy_loss_scale=0.0,
        reinforce_loss_scale=0.0,
        dynamics_backprop_loss_scale=1.0,
        actor_entropy_loss_schedule="0.0",
        adam_eps=1e-7,
        weight_decay=0.0,
        soft_target_tau=1,
        target_update_period=100,
        use_pred_discount=True,
        plotter=None,
        render_eval_paths=False,
        debug=False,
    ):
        super().__init__()

        torch.autograd.set_detect_anomaly(debug)

        torch.backends.cudnn.benchmark = True

        self.env = env
        self.actor = actor.to(ptu.device)
        self.world_model = world_model.to(ptu.device)
        self.vf = vf.to(ptu.device)

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        if optimizer_class == "torch_adam":
            optimizer_class = optim.Adam
        elif optimizer_class == "apex_adam" and APEX_AVAILABLE:
            optimizer_class = apex.optimizers.FusedAdam

        self.actor_optimizer = optimizer_class(
            self.actor.parameters(),
            lr=actor_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        self.world_model_optimizer = optimizer_class(
            self.world_model.parameters(),
            lr=world_model_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        self.use_amp = use_amp and APEX_AVAILABLE
        if self.use_amp:
            models, optimizers = amp.initialize(
                [
                    self.world_model.img_step_layer,
                    self.world_model.img_step_mlp,
                    self.world_model.obs_step_mlp,
                    self.world_model.conv_decoder,
                    self.world_model.conv_encoder,
                    self.world_model.pred_discount,
                    self.world_model.reward,
                    self.world_model.rnn,
                    self.actor,
                    self.vf,
                ],
                [self.world_model_optimizer, self.actor_optimizer, self.vf_optimizer],
                opt_level=opt_level,
                num_losses=3,
            )
            (
                self.world_model.img_step_layer,
                self.world_model.img_step_mlp,
                self.world_model.obs_step_mlp,
                self.world_model.conv_decoder,
                self.world_model.conv_encoder,
                self.world_model.pred_discount,
                self.world_model.reward,
                self.world_model.rnn,
                self.actor,
                self.vf,
            ) = models
            (
                self.world_model_optimizer,
                self.actor_optimizer,
                self.vf_optimizer,
            ) = optimizers

        self.opt_level = opt_level
        self.discount = discount
        self.reward_scale = reward_scale
        self.gradient_clip = gradient_clip
        self.lam = lam
        self.imagination_horizon = imagination_horizon
        self.free_nats = free_nats
        self.kl_loss_scale = kl_loss_scale
        self.pred_discount_loss_scale = pred_discount_loss_scale
        self.image_loss_scale = image_loss_scale
        self.reward_loss_scale = reward_loss_scale
        self.transition_loss_scale = transition_loss_scale
        self.entropy_loss_scale = entropy_loss_scale
        self.reinforce_loss_scale = reinforce_loss_scale
        self.dynamics_backprop_loss_scale = dynamics_backprop_loss_scale
        self.actor_entropy_loss_scale = lambda x=actor_entropy_loss_schedule: schedule(
            x, self._n_train_steps_total
        )
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.target_vf = target_vf
        self.use_pred_discount = use_pred_discount
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def try_update_target_networks(self):
        if (
            self.target_vf
            and self._n_train_steps_total % self.target_update_period == 0
        ):
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def train_from_torch(self, batch):
        gt.blank_stamp()
        self.try_update_target_networks()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """

        self._n_train_steps_total += 1
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp("dreamer training", unique=False)

    def imagine_ahead(self, state):
        new_state = {}
        for k, v in state.items():
            with torch.no_grad():
                if self.use_pred_discount:  # Last step could be terminal.
                    v = v[:, :-1]
                new_state[k] = torch.cat([v[:, i, :] for i in range(v.shape[1])])
        feats = []
        actions = []
        for i in range(self.imagination_horizon):
            feat = self.world_model.get_feat(new_state).detach()
            action = self.actor(feat).rsample()
            new_state = self.world_model.img_step(new_state, action)
            feats.append(self.world_model.get_feat(new_state).unsqueeze(0))
            actions.append(action.unsqueeze(0))
        feats = torch.cat(feats)
        actions = torch.cat(actions)
        return feats, actions

    def world_model_loss(
        self,
        image_dist,
        reward_dist,
        prior,
        post,
        prior_dist,
        post_dist,
        pred_discount_dist,
        obs,
        rewards,
        terminals,
    ):
        image_pred_loss = (
            -1
            * image_dist.log_prob(
                self.world_model.preprocess(obs).reshape(-1, 3, 64, 64)
            ).mean()
        )
        reward_pred_loss = -1 * reward_dist.log_prob(rewards).mean()
        pred_discount_target = self.discount * (1 - terminals.float())
        pred_discount_loss = (
            -1 * pred_discount_dist.log_prob(pred_discount_target).mean()
        )
        div = kld(post_dist, prior_dist).mean()
        div = torch.max(div, ptu.tensor(self.free_nats))
        post_detached_dist = self.world_model.get_detached_dist(
            post["mean"], post["std"]
        )
        prior_detached_dist = self.world_model.get_detached_dist(
            prior["mean"], prior["std"]
        )
        prior_kld = kld(prior_dist, post_detached_dist).mean()
        post_kld = kld(prior_detached_dist, post_dist).mean()
        transition_loss = torch.max(prior_kld, ptu.tensor(self.free_nats))
        entropy_loss = torch.max(post_kld, ptu.tensor(self.free_nats))
        world_model_loss = (
            self.kl_loss_scale * div
            + self.image_loss_scale * image_pred_loss
            + self.reward_loss_scale * reward_pred_loss
            + self.transition_loss_scale * transition_loss
            + self.entropy_loss_scale * entropy_loss
        )

        if self.use_pred_discount:
            world_model_loss += self.pred_discount_loss_scale * pred_discount_loss
        return (
            world_model_loss,
            div,
            image_pred_loss,
            reward_pred_loss,
            transition_loss,
            entropy_loss,
            pred_discount_loss,
        )

    def actor_loss(self, imag_returns, value, imag_feat, imag_actions, weights):
        assert len(imag_returns.shape) == 3, imag_returns.shape
        assert len(value.shape) == 3, value.shape
        assert len(imag_feat.shape) == 3, imag_feat.shape
        assert len(imag_actions.shape) == 3, imag_actions.shape
        assert len(weights.shape) == 3 and weights.shape[-1] == 1, weights.shape

        baseline_shifted_returns = imag_returns - value[:-1]
        baseline_shifted_returns = (
            torch.cat(
                [
                    baseline_shifted_returns[i, :, :]
                    for i in range(baseline_shifted_returns.shape[0])
                ]
            )
            .detach()
            .squeeze(-1)
        )

        imag_feat_a = torch.cat(
            [imag_feat[i, :, :] for i in range(imag_feat.shape[0] - 1)]
        ).detach()

        imag_actions = torch.cat(
            [imag_actions[i, :, :] for i in range(imag_actions.shape[0] - 1)]
        ).detach()
        assert imag_actions.shape[0] == imag_feat_a.shape[0]

        imag_actor_dist = self.actor(imag_feat_a)
        log_probs = imag_actor_dist.log_prob(imag_actions)
        reinforce_loss = -1 * log_probs * baseline_shifted_returns
        actor_entropy_loss = -1 * imag_actor_dist.entropy()

        dynamics_backprop_loss = -(imag_returns)
        dynamics_backprop_loss = torch.cat(
            [
                dynamics_backprop_loss[i, :, :]
                for i in range(dynamics_backprop_loss.shape[0])
            ]
        ).squeeze(-1)
        weights = torch.cat(
            [weights[i, :, :] for i in range(weights.shape[0])]
        ).squeeze(-1)
        assert (
            dynamics_backprop_loss.shape
            == reinforce_loss.shape
            == actor_entropy_loss.shape
            == weights.shape
        )
        actor_entropy_loss_scale = self.actor_entropy_loss_scale()
        actor_loss = (
            (
                self.dynamics_backprop_loss_scale * dynamics_backprop_loss
                + self.reinforce_loss_scale * reinforce_loss
                + actor_entropy_loss_scale * actor_entropy_loss
            )
            * weights
        ).mean()
        return (
            actor_loss,
            dynamics_backprop_loss.mean(),
            reinforce_loss.mean(),
            actor_entropy_loss.mean(),
            actor_entropy_loss_scale,
        )

    def value_loss(self, imag_feat_v, weights, imag_returns):
        assert len(imag_feat_v.shape) == 3, imag_feat_v.shape
        assert len(weights.shape) == 3 and weights.shape[-1] == 1, weights.shape
        assert len(imag_returns.shape) == 3, imag_returns.shape

        value_dist = self.world_model.get_dist(self.vf(imag_feat_v)[:-1], 1)
        log_probs = value_dist.log_prob(imag_returns)
        weights = weights.squeeze(-1)
        assert len(log_probs.shape) == 2, log_probs.shape
        assert weights.shape == log_probs.shape

        # squeeze to get the last dimension of weights to go away (it is 1 anyways)
        vf_loss = -(weights * log_probs).mean()
        return vf_loss, value_dist

    def update_network(self, network, optimizer, loss, loss_id):
        zero_grad(network)
        if self.use_amp:
            with amp.scale_loss(loss, optimizer, loss_id=loss_id) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if self.gradient_clip > 0:
            if not self.use_amp:
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), self.gradient_clip, norm_type=2
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer),
                    self.gradient_clip,
                    norm_type=2,
                )
        optimizer.step()

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
        **kwargs,
    ) -> Tuple[DreamerLosses, LossStatistics]:
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        """
        World Model Loss
        """
        (
            post,
            prior,
            post_dist,
            prior_dist,
            image_dist,
            reward_dist,
            pred_discount_dist,
            embed,
        ) = self.world_model(obs, actions)

        # stack obs, rewards and terminals along path dimension
        obs = torch.cat([obs[:, i, :] for i in range(obs.shape[1])])
        rewards = torch.cat([rewards[:, i, :] for i in range(rewards.shape[1])])
        terminals = torch.cat([terminals[:, i, :] for i in range(terminals.shape[1])])

        (
            world_model_loss,
            div,
            image_pred_loss,
            reward_pred_loss,
            transition_loss,
            entropy_loss,
            pred_discount_loss,
        ) = self.world_model_loss(
            image_dist,
            reward_dist,
            prior,
            post,
            prior_dist,
            post_dist,
            pred_discount_dist,
            obs,
            rewards,
            terminals,
        )

        self.update_network(
            self.world_model, self.world_model_optimizer, world_model_loss, 0
        )

        """
        Actor Loss
        """
        world_model_params = list(self.world_model.parameters())
        vf_params = list(self.vf.parameters())
        pred_discount_params = list(self.world_model.pred_discount.parameters())
        with FreezeParameters(world_model_params):
            imag_feat, imag_actions = self.imagine_ahead(post)
        with FreezeParameters(world_model_params + vf_params):
            imag_reward = self.world_model.reward(imag_feat)
            if self.use_pred_discount:
                with FreezeParameters(pred_discount_params):
                    discount = self.world_model.get_dist(
                        self.world_model.pred_discount(imag_feat),
                        std=None,
                        normal=False,
                    ).mean
            else:
                discount = self.discount * torch.ones_like(imag_reward)
            if self.target_vf:
                with FreezeParameters(self.target_vf.modules):
                    value = self.target_vf(imag_feat)
            else:
                value = self.vf(imag_feat)
        imag_returns = lambda_return(
            imag_reward[:-1],
            value[:-1],
            discount[:-1],
            bootstrap=value[-1],
            lambda_=self.lam,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()[:-1]
        (
            actor_loss,
            dynamics_backprop_loss,
            reinforce_loss,
            actor_entropy_loss,
            actor_entropy_loss_scale,
        ) = self.actor_loss(imag_returns, value, imag_feat, imag_actions, weights)

        self.update_network(self.actor, self.actor_optimizer, actor_loss, 1)

        """
        Value Loss
        """
        with torch.no_grad():
            imag_feat_v = imag_feat.detach()
            target = imag_returns.detach()
            weights = weights.detach()

        vf_loss, value_dist = self.value_loss(imag_feat_v, weights, target)

        self.update_network(self.vf, self.vf_optimizer, vf_loss, 2)

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics["World Model Loss"] = world_model_loss.item()
            eval_statistics["Image Loss"] = image_pred_loss.item()
            eval_statistics["Reward Loss"] = reward_pred_loss.item()
            eval_statistics["Divergence Loss"] = div.item()
            eval_statistics["Transition Loss"] = transition_loss.item()
            eval_statistics["Entropy Loss"] = entropy_loss.item()
            if self.use_pred_discount:
                eval_statistics["Pred Discount Loss"] = pred_discount_loss.item()
            eval_statistics["Value Loss"] = vf_loss.item()
            eval_statistics["Actor Loss"] = actor_loss.item()
            eval_statistics["Dynamics Backprop Loss"] = dynamics_backprop_loss.item()
            eval_statistics["Reinforce Loss"] = reinforce_loss.item()
            eval_statistics["Actor Entropy Loss"] = actor_entropy_loss.item()
            eval_statistics["Actor Entropy Loss Scale"] = actor_entropy_loss_scale
            eval_statistics["Imagined Returns"] = imag_returns.mean().item()
            eval_statistics["Imagined Rewards"] = imag_reward.mean().item()
            eval_statistics["Imagined Values"] = value_dist.mean.mean().item()
            eval_statistics["Predicted Rewards"] = reward_dist.mean.mean().item()

        loss = DreamerLosses(
            actor_loss=actor_loss,
            world_model_loss=world_model_loss,
            vf_loss=vf_loss,
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
