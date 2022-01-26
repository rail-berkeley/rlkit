import os.path as osp
from collections import Counter, OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.distributions import kl_divergence as kld

import rlkit.torch.pytorch_util as ptu
from rlkit.core.loss import LossFunction, LossStatistics
from rlkit.torch.model_based.dreamer.utils import (
    FreezeParameters,
    lambda_return,
    schedule,
)
from rlkit.torch.torch_rl_algorithm import TorchTrainer

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
        target_vf,
        world_model,
        image_shape,
        imagination_horizon=15,
        discount=0.99,
        actor_lr=8e-5,
        vf_lr=8e-5,
        world_model_lr=3e-4,
        world_model_gradient_clip=100.0,
        actor_gradient_clip=100.0,
        value_gradient_clip=100.0,
        adam_eps=1e-5,
        weight_decay=0.0,
        soft_target_tau=1,
        target_update_period=100,
        lam=0.95,
        free_nats=1.0,
        kl_loss_scale=0.0,
        pred_discount_loss_scale=10.0,
        image_loss_scale=1.0,
        reward_loss_scale=2.0,
        transition_loss_scale=0.8,
        detach_rewards=False,
        forward_kl=False,
        policy_gradient_loss_scale=0.0,
        actor_entropy_loss_schedule="1e-4",
        use_pred_discount=False,
        reward_scale=1,
        num_imagination_iterations=1,
        use_baseline=True,
        use_ppo_loss=False,
        ppo_clip_param=0.2,
        num_actor_value_updates=1,
        use_advantage_normalization=False,
        use_clipped_value_loss=False,
        actor_value_lr=8e-5,
        use_actor_value_optimizer=False,
    ):
        super().__init__()

        torch.backends.cudnn.benchmark = True

        self.scaler = torch.cuda.amp.GradScaler()
        self.env = env
        self.use_pred_discount = use_pred_discount
        self.actor = actor.to(ptu.device)
        self.world_model = world_model.to(ptu.device)
        self.vf = vf.to(ptu.device)
        self.target_vf = target_vf.to(ptu.device)

        self.optimizer_class = optim.Adam

        self.actor_lr = actor_lr
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.vf_lr = vf_lr
        self.world_model_lr = world_model_lr

        self.actor_optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=actor_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        self.vf_optimizer = self.optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        self.world_model_optimizer = self.optimizer_class(
            self.world_model.parameters(),
            lr=world_model_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        self.use_actor_value_optimizer = use_actor_value_optimizer
        self.actor_value_optimizer = self.optimizer_class(
            list(self.actor.parameters()) + list(self.vf.parameters()),
            lr=actor_value_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )

        self.discount = discount
        self.lam = lam
        self.imagination_horizon = imagination_horizon
        self.free_nats = free_nats
        self.kl_loss_scale = kl_loss_scale
        self.pred_discount_loss_scale = pred_discount_loss_scale
        self.image_loss_scale = image_loss_scale
        self.reward_loss_scale = reward_loss_scale
        self.transition_loss_scale = transition_loss_scale
        self.policy_gradient_loss_scale = policy_gradient_loss_scale
        self.actor_entropy_loss_scale = lambda x=actor_entropy_loss_schedule: schedule(
            x, self._n_train_steps_total
        )
        self.forward_kl = forward_kl
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.image_shape = image_shape
        self.use_baseline = use_baseline
        self.use_ppo_loss = use_ppo_loss
        self.ppo_clip_param = ppo_clip_param
        self.num_actor_value_updates = num_actor_value_updates
        self.world_model_gradient_clip = world_model_gradient_clip
        self.actor_gradient_clip = actor_gradient_clip
        self.value_gradient_clip = value_gradient_clip
        self.use_advantage_normalization = use_advantage_normalization
        self.detach_rewards = detach_rewards
        self.num_imagination_iterations = num_imagination_iterations
        self.use_clipped_value_loss = use_clipped_value_loss
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def train_from_torch(self, batch):
        self.try_update_target_networks()
        _, stats = self.train_networks(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )

        self._n_train_steps_total += 1
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            self._need_to_update_eval_statistics = False

    def imagine_ahead(self, state, actor=None):
        if actor is None:
            actor = self.actor
        new_state = {}
        for k, v in state.items():
            if self.use_pred_discount:
                v = v[:, :-1]
            if k == "stoch" and self.world_model.discrete_latents:
                new_state[k] = v.reshape(-1, v.shape[-2], v.shape[-1])
            else:
                new_state[k] = v.reshape(-1, v.shape[-1]).detach()
        imagined_features = []
        imagined_actions = []
        states = dict(mean=[], std=[], stoch=[], deter=[])
        for _ in range(self.imagination_horizon):
            features = self.world_model.get_features(new_state)
            for k in states.keys():
                states[k].append(new_state[k].unsqueeze(0))
            action_dist = actor(features.detach())
            action = action_dist.rsample()
            new_state = self.world_model.action_step(new_state, action)

            imagined_features.append(features.unsqueeze(0))
            imagined_actions.append(action.unsqueeze(0))
        imagined_features = torch.cat(imagined_features)
        imagined_actions = torch.cat(imagined_actions)
        for k in states.keys():
            states[k] = torch.cat(states[k])
        return imagined_features, imagined_actions, states

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
        """
        :param image_dist: Normal(mu, 1)
            mu: torch.Tensor (batch_size*(path_length+1), image_shape)
        :param reward_dist: Normal(mu, 1)
            mu: torch.Tensor (batch_size*(path_length+1), 1)
        :param prior: dict
            stoch: torch.Tensor (batch_size*(path_length+1), latent_size)
            deter: torch.Tensor (batch_size*(path_length+1), latent_size)
            mean: torch.Tensor (batch_size*(path_length+1), latent_size)
            std: torch.Tensor (batch_size*(path_length+1), latent_size)
        :param post: dict
            stoch: torch.Tensor (batch_size*(path_length+1), latent_size)
            deter: torch.Tensor (batch_size*(path_length+1), latent_size)
            mean: torch.Tensor (batch_size*(path_length+1), latent_size)
            std: torch.Tensor (batch_size*(path_length+1), latent_size)
        :param prior_dist: Normal(prior['mean'], prior['std'])
        :param post_dist: Normal(post['mean'], post['std'])
        :param pred_discount_dist: Bernoulli(logits, 1)
            logits: torch.Tensor (batch_size*(path_length+1), 1)
        :param obs: torch.Tensor (batch_size*(path_length+1), image_shape)
        :param rewards: torch.Tensor (batch_size*(path_length+1), 1)
        :param terminals: torch.Tensor (batch_size*(path_length+1), 1)
        :return losses (list):
            world_model_loss,
            div,
            image_pred_loss,
            reward_pred_loss,
            transition_loss,
            entropy_loss,
            pred_discount_loss,
        """
        preprocessed_obs = self.world_model.preprocess(obs)
        image_pred_loss = -1 * image_dist.log_prob(preprocessed_obs).mean()
        if self.detach_rewards:
            reward_pred_loss = -1 * reward_dist.log_prob(rewards.detach()).mean()
        else:
            reward_pred_loss = -1 * reward_dist.log_prob(rewards).mean()
        pred_discount_target = self.discount * (1 - terminals.float())
        pred_discount_loss = (
            -1 * pred_discount_dist.log_prob(pred_discount_target).mean()
        )

        post_detached_dist = self.world_model.get_detached_dist(post)
        prior_detached_dist = self.world_model.get_detached_dist(prior)
        if self.forward_kl:
            div = kld(post_dist, prior_dist).mean()
            div = torch.max(div, ptu.tensor(self.free_nats))
            prior_kld = kld(post_detached_dist, prior_dist).mean()
            post_kld = kld(post_dist, prior_detached_dist).mean()
        else:
            div = kld(prior_dist, post_dist).mean()
            div = torch.max(div, ptu.tensor(self.free_nats))
            prior_kld = kld(prior_dist, post_detached_dist).mean()
            post_kld = kld(prior_detached_dist, post_dist).mean()
        transition_loss = torch.max(prior_kld, ptu.tensor(self.free_nats))
        entropy_loss = torch.max(post_kld, ptu.tensor(self.free_nats))
        entropy_loss_scale = 1 - self.transition_loss_scale
        entropy_loss_scale = (1 - self.kl_loss_scale) * entropy_loss_scale
        transition_loss_scale = (1 - self.kl_loss_scale) * self.transition_loss_scale
        world_model_loss = (
            self.kl_loss_scale * div
            + self.image_loss_scale * image_pred_loss
            + self.reward_loss_scale * reward_pred_loss
            + transition_loss_scale * transition_loss
            + entropy_loss_scale * entropy_loss
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

    def actor_loss(
        self,
        imagined_returns,
        value,
        imagined_features,
        imagined_actions,
        weights,
        old_imagined_log_probs,
        actor,
        log_keys,
        prefix="",
    ):
        """
        :param imagined_returns:
        :param value:
        :param imagined_features:
        :param imagined_actions:
        :param weights:
        :param old_imagined_log_probs:
        :param actor:
        :param log_keys:
        :param prefix:

        :return actor_loss:
        """
        if self.use_baseline:
            advantages = imagined_returns - value[:-1]
        else:
            advantages = imagined_returns
        if self.use_advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        advantages = advantages.reshape(-1).detach()

        imagined_features_actions = (
            imagined_features[:-1].reshape(-1, imagined_features.shape[-1]).detach()
        )
        imagined_actions = (
            imagined_actions[:-1].reshape(-1, imagined_actions.shape[-1]).detach()
        )

        imagined_actor_dist = actor(imagined_features_actions)
        imagined_log_probs = imagined_actor_dist.log_prob(imagined_actions)
        if self.use_ppo_loss:
            ratio = torch.exp(imagined_log_probs - old_imagined_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.ppo_clip_param, 1.0 + self.ppo_clip_param)
                * advantages
            )
            policy_gradient_loss = -torch.min(surr1, surr2)
        else:
            policy_gradient_loss = -1 * imagined_log_probs * advantages
        actor_entropy_loss = -1 * imagined_actor_dist.entropy()

        dynamics_backprop_loss = -(imagined_returns)
        dynamics_backprop_loss = dynamics_backprop_loss.reshape(-1)
        weights = weights.reshape(-1)
        actor_entropy_loss_scale = self.actor_entropy_loss_scale()
        dynamics_backprop_loss_scale = 1 - self.policy_gradient_loss_scale
        if self.num_actor_value_updates > 1:
            actor_loss = (
                (
                    self.policy_gradient_loss_scale * policy_gradient_loss
                    + actor_entropy_loss_scale * actor_entropy_loss
                )
                * weights
            ).mean()
        else:
            actor_loss = (
                (
                    dynamics_backprop_loss_scale * dynamics_backprop_loss
                    + self.policy_gradient_loss_scale * policy_gradient_loss
                    + actor_entropy_loss_scale * actor_entropy_loss
                )
                * weights
            ).mean()
        log_keys[prefix + "actor_loss"] += actor_loss.item()
        log_keys[
            prefix + "dynamics_backprop_loss"
        ] += dynamics_backprop_loss.mean().item()
        log_keys[prefix + "actor_entropy_loss"] += actor_entropy_loss.mean().item()
        log_keys[prefix + "actor_entropy_loss_scale"] += actor_entropy_loss_scale
        log_keys[prefix + "imagined_log_probs"] += imagined_log_probs.mean().item()
        log_keys[prefix + "policy_gradient_loss"] += policy_gradient_loss.mean().item()
        log_keys[prefix + "log_probs"] += imagined_log_probs.mean().item()
        return actor_loss

    def value_loss(
        self,
        imagined_features_v,
        weights,
        imagined_returns,
        vf,
        log_keys,
        old_imagined_value=None,
        prefix="",
    ):
        """
        :param imagined_features_v:
        :param weights:
        :param imagined_returns:
        :param vf:
        :param log_keys:
        :param old_imagined_value:
        :param prefix:

        :return vf_loss:
        """
        values = vf(imagined_features_v)[:-1]
        value_dist = self.world_model.get_dist(values, 1)
        if self.use_clipped_value_loss:
            value_pred_clipped = old_imagined_value[:-1] + (
                values - old_imagined_value[:-1]
            ).clamp(-self.ppo_clip_param, self.ppo_clip_param)
            vf_losses = value_dist.log_prob(imagined_returns)
            vf_clipped_dist = self.world_model.get_dist(value_pred_clipped, 1)
            vf_losses_clipped = vf_clipped_dist.log_prob(imagined_returns)
            weights = weights.squeeze(-1)
            vf_loss = (torch.max(-vf_losses, -vf_losses_clipped) * weights).mean()
        else:
            log_probs = value_dist.log_prob(imagined_returns)
            weights = weights.squeeze(-1)
            vf_loss = -(weights * log_probs).mean()
        log_keys[prefix + "value_loss"] += vf_loss.item()
        log_keys[prefix + "imagined_value_mean"] += value_dist.mean.mean().item()
        return vf_loss

    def update_network(self, network, optimizer, loss, gradient_clip):
        if isinstance(network, list):
            parameters = []
            for net in network:
                parameters.extend(list(net.parameters()))
        else:
            parameters = list(network.parameters())
        if gradient_clip > 0:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, norm_type=2)
        self.scaler.step(optimizer)
        optimizer.zero_grad(set_to_none=True)
        self.scaler.update()

    def train_networks(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[DreamerLosses, LossStatistics]:
        """
        :param: batch dict[
            "rewards": (batch_size, path_length+1, 1),
            "terminals": (batch_size, path_length+1, 1),
            "observations": (batch_size, path_length+1, obs_dim),
            "actions": (batch_size, path_length+1, act_dim),
            ]
        :param: skip_statistics bool
        :returns: Tuple[DreamerLosses, LossStatistics]
        """

        """
        World Model Loss
        """
        rewards = batch["rewards"] * self.reward_scale
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        with torch.cuda.amp.autocast():
            (
                post,
                prior,
                post_dist,
                prior_dist,
                image_dist,
                reward_dist,
                pred_discount_dist,
                _,
            ) = self.world_model(obs, actions)
            obs = obs.reshape(-1, *self.image_shape)
            rewards = rewards.reshape(-1, rewards.shape[-1])
            terminals = terminals.reshape(-1, terminals.shape[-1])
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
        self.scaler.scale(world_model_loss).backward()
        self.update_network(
            self.world_model,
            self.world_model_optimizer,
            world_model_loss,
            self.world_model_gradient_clip,
        )

        """
        Actor Value Loss
        """
        world_model_params = list(self.world_model.parameters())
        vf_params = list(self.vf.parameters())
        target_vf_params = list(self.target_vf.parameters())
        pred_discount_params = list(self.world_model.pred_discount.parameters())
        log_keys = Counter()
        for _ in range(self.num_imagination_iterations):
            with torch.cuda.amp.autocast():
                with FreezeParameters(
                    world_model_params + pred_discount_params + vf_params
                ):
                    (imagined_features, imagined_actions, _) = self.imagine_ahead(post)
                    imagined_reward = self.world_model.reward(imagined_features)
                    if self.use_pred_discount:
                        discount = self.world_model.get_dist(
                            self.world_model.pred_discount(imagined_features),
                            std=None,
                            normal=False,
                        ).mean
                    else:
                        discount = self.discount * torch.ones_like(imagined_reward)
                    old_imagined_value = self.vf(imagined_features).detach()
                imagined_features_actions = (
                    imagined_features[:-1]
                    .reshape(-1, imagined_features.shape[-1])
                    .detach()
                )
                imagined_actions_actions = (
                    imagined_actions[:-1]
                    .reshape(-1, imagined_actions.shape[-1])
                    .detach()
                )
                imagined_actor_dist = self.actor(imagined_features_actions)
                imagined_log_probs = imagined_actor_dist.log_prob(
                    imagined_actions_actions
                ).detach()
            for _ in range(self.num_actor_value_updates):
                with torch.cuda.amp.autocast():
                    with FreezeParameters(vf_params + target_vf_params):
                        imagined_target_value = self.target_vf(imagined_features)
                        imagined_value = self.vf(imagined_features)
                    imagined_returns = lambda_return(
                        imagined_reward[:-1],
                        imagined_target_value[:-1],
                        discount[:-1],
                        bootstrap=imagined_target_value[-1],
                        lambda_=self.lam,
                    )
                    weights = torch.cumprod(
                        torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
                    ).detach()[:-1]

                    actor_loss = self.actor_loss(
                        imagined_returns,
                        imagined_value,
                        imagined_features,
                        imagined_actions,
                        weights,
                        imagined_log_probs,
                        self.actor,
                        log_keys,
                    )

                    with torch.no_grad():
                        imagined_features_values = imagined_features.detach()
                        target = imagined_returns.detach()
                        weights = weights.detach()

                    vf_loss = self.value_loss(
                        imagined_features_values,
                        weights,
                        target,
                        self.vf,
                        log_keys,
                        old_imagined_value,
                    )

                if self.use_actor_value_optimizer:
                    self.scaler.scale(actor_loss + vf_loss).backward()
                    self.update_network(
                        [self.actor, self.vf],
                        self.actor_value_optimizer,
                        actor_loss + vf_loss,
                        self.actor_gradient_clip,
                    )
                else:
                    self.scaler.scale(actor_loss).backward()
                    self.update_network(
                        self.actor,
                        self.actor_optimizer,
                        actor_loss,
                        self.actor_gradient_clip,
                    )
                    self.scaler.scale(vf_loss).backward()
                    self.update_network(
                        self.vf,
                        self.vf_optimizer,
                        vf_loss,
                        self.value_gradient_clip,
                    )

        if self.num_imagination_iterations > 0:
            for key in log_keys:
                log_keys[key] /= (
                    self.num_actor_value_updates * self.num_imagination_iterations
                )
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
            eval_statistics["Pred Discount Loss"] = pred_discount_loss.item()
            if not self.world_model.discrete_latents:
                eval_statistics["Posterior State Std"] = post["std"].mean().item()
                eval_statistics["Prior State Std"] = prior["std"].mean().item()
            eval_statistics["Pred Discount Loss"] = pred_discount_loss.item()

            eval_statistics["Actor Loss"] = log_keys["actor_loss"]
            eval_statistics["Actor Entropy"] = -1 * log_keys["actor_entropy_loss"]
            eval_statistics["Dynamics Backprop Loss"] = log_keys[
                "dynamics_backprop_loss"
            ]
            eval_statistics["Policy Gradient Loss"] = log_keys["policy_gradient_loss"]
            eval_statistics["Actor Entropy Loss"] = log_keys["actor_entropy_loss"]
            eval_statistics["Actor Entropy Loss Scale"] = log_keys[
                "actor_entropy_loss_scale"
            ]
            eval_statistics["Actor Log Probs"] = log_keys["log_probs"]
            eval_statistics["Value Loss"] = log_keys["value_loss"]

            if self.num_imagination_iterations > 0:
                eval_statistics["Imagined Returns"] = imagined_returns.mean().item()
                eval_statistics["Imagined Rewards"] = imagined_reward.mean().item()
                eval_statistics["Imagined Values"] = log_keys["imagined_values_mean"]
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
            self.target_vf,
        ]

    @property
    def optimizers(self):
        return [
            self.actor_optimizer,
            self.vf_optimizer,
            self.world_model_optimizer,
        ]

    def get_snapshot(self):
        return dict()

    def compute_loss(self, batch, skip_statistics):
        pass

    @torch.no_grad()
    def evaluate(self, batch, buffer_data=True):
        if buffer_data:
            rewards = ptu.from_numpy(batch["rewards"])
            terminals = ptu.from_numpy(batch["terminals"])
            obs = ptu.from_numpy(batch["observations"])
            actions = ptu.from_numpy(batch["actions"])
            prefix = "replay/"
        else:
            rewards = ptu.from_numpy(batch["rewards"])
            terminals = ptu.from_numpy(batch["terminals"])
            obs = batch["observations"]
            actions = batch["high_level_actions"][:, :, :-1]
            rt_idxs = np.arange(
                10,
                obs.shape[1],
                10,
            )
            rt_idxs = np.concatenate([[0], rt_idxs])
            obs = obs[:, rt_idxs]
            actions = actions[:, rt_idxs]
            prefix = "validation/"
            obs = ptu.from_numpy(obs)
            actions = ptu.from_numpy(actions)
        with torch.cuda.amp.autocast():
            (
                post,
                prior,
                post_dist,
                prior_dist,
                image_dist,
                reward_dist,
                pred_discount_dist,
                _,
            ) = self.world_model(obs, actions)
            ob = obs.reshape(-1, *self.image_shape)
            rewards = rewards.reshape(-1, rewards.shape[-1])
            terminals = terminals.reshape(-1, terminals.shape[-1])
            (_, _, image_pred_loss, _, _, _, _,) = self.world_model_loss(
                image_dist,
                reward_dist,
                prior,
                post,
                prior_dist,
                post_dist,
                pred_discount_dist,
                ob,
                rewards,
                terminals,
            )
        reward_error = torch.nn.functional.mse_loss(rewards, reward_dist.mean).item()
        discount_error = torch.nn.functional.mse_loss(
            pred_discount_dist.mean, terminals
        ).item()

        self.eval_statistics[prefix + "Full Obs Image Loss"] = image_pred_loss.item()
        self.eval_statistics[prefix + "Full Obs Reward Error"] = reward_error
        self.eval_statistics[prefix + "Full Obs Discount Error"] = discount_error

        self.eval_statistics[
            prefix + "Full Obs Primitive Model Image Loss"
        ] = image_pred_loss.item()
        self.eval_statistics[prefix + "Full Obs Primitive Reward Error"] = reward_error
        self.eval_statistics[
            prefix + "Full Obs Primitive Discount Error"
        ] = discount_error

        self.eval_statistics[prefix + "RAPS Obs Image Loss"] = image_pred_loss.item()
        self.eval_statistics[prefix + "RAPS Obs Reward Error"] = reward_error
        self.eval_statistics[prefix + "RAPS Obs Discount Error"] = discount_error

        self.eval_statistics[
            prefix + "RAPS Obs Primitive Model Image Loss"
        ] = image_pred_loss.item()
        self.eval_statistics[prefix + "RAPS Obs Primitive Reward Error"] = reward_error
        self.eval_statistics[
            prefix + "RAPS Obs Primitive Discount Error"
        ] = discount_error

    def save(self, filepath):
        torch.save(self.actor.state_dict(), osp.join(filepath, "actor.ptc"))
        torch.save(self.vf.state_dict(), osp.join(filepath, "vf.ptc"))
        torch.save(self.target_vf.state_dict(), osp.join(filepath, "target_vf.ptc"))
        torch.save(self.world_model.state_dict(), osp.join(filepath, "world_model.ptc"))


class DreamerV2LowLevelRAPSTrainer(DreamerV2Trainer):
    def __init__(
        self,
        *args,
        num_primitives,
        num_low_level_actions_per_primitive=100,
        batch_length=50,
        binarize_rewards=False,
        num_world_model_training_iterations=1,
        wm_loss_scale=1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_primitives = num_primitives
        self.num_low_level_actions_per_primitive = num_low_level_actions_per_primitive
        self.batch_length = batch_length
        self.criterion = nn.MSELoss()
        self.train_wm = True
        self.binarize_rewards = binarize_rewards
        self.num_world_model_training_iterations = num_world_model_training_iterations
        self.wm_loss_scale = wm_loss_scale

    def imagine_ahead(self, state, actor=None):
        if actor is None:
            actor = self.actor
        new_state = {}
        for k, v in state.items():
            if self.use_pred_discount:
                v = v[:, :-1]
            if k == "stoch" and self.world_model.discrete_latents:
                new_state[k] = v.reshape(-1, v.shape[-2], v.shape[-1]).detach()
            else:
                new_state[k] = v.reshape(-1, v.shape[-1]).detach()
        imagined_features = []
        imagined_actions = []
        states = dict(mean=[], std=[], stoch=[], deter=[])
        for _ in range(self.imagination_horizon):
            features = self.world_model.get_features(new_state)
            for k in states.keys():
                states[k].append(new_state[k].unsqueeze(0))
            action_dist = actor(features.detach())
            high_level_action = action_dist.rsample()
            new_state = self.world_model.forward_high_level_step_primitive_model(
                new_state, high_level_action, self.num_low_level_actions_per_primitive
            )
            imagined_features.append(features.unsqueeze(0))
            imagined_actions.append(high_level_action.unsqueeze(0))
        imagined_features = torch.cat(imagined_features)
        imagined_actions = torch.cat(imagined_actions)
        for k in states.keys():
            states[k] = torch.cat(states[k])
        return imagined_features, imagined_actions, states

    def train_networks(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[DreamerLosses, LossStatistics]:
        """
        :param: batch dict[
            "rewards": (batch_size, path_length+1, 1),
            "terminals": (batch_size, path_length+1, 1),
            "observations": (batch_size, path_length*num_low_level_actions_per_primitive+1, obs_dim),
            "high_level_actions": (batch_size, path_length*num_low_level_actions_per_primitive+1, act_dim),
            "low_level_actions": (batch_size, path_length*num_low_level_actions_per_primitive+1, low_level_act_dim),
            ]
        :param: skip_statistics bool
        :returns: Tuple[DreamerLosses, LossStatistics]
        """

        """
        World Model Loss
        """

        with torch.cuda.amp.autocast():
            for itr in range(self.num_world_model_training_iterations):
                batch = self.buffer.random_batch(batch["rewards"].shape[0])
                rewards = ptu.from_numpy(batch["rewards"])
                terminals = ptu.from_numpy(batch["terminals"])
                obs = ptu.from_numpy(batch["observations"])
                high_level_actions = ptu.from_numpy(batch["high_level_actions"])
                low_level_actions = ptu.from_numpy(batch["low_level_actions"])
                with torch.cuda.amp.autocast():
                    rt_idxs = np.arange(
                        self.num_low_level_actions_per_primitive,
                        obs.shape[1],
                        self.num_low_level_actions_per_primitive,
                    )
                    # reset obs, effect of first primitive, second primitive, so on
                    rt_idxs = np.concatenate([[0], rt_idxs])

                    batch_start = np.random.randint(
                        0, obs.shape[1] - self.batch_length + 1, size=(obs.shape[0])
                    )
                    batch_indices = (
                        np.linspace(
                            batch_start,
                            batch_start + self.batch_length,
                            self.batch_length,
                            endpoint=False,
                        )
                        .astype(int)
                        .transpose(1, 0)
                    )
                    (
                        post,
                        prior,
                        post_dist,
                        prior_dist,
                        image_dist,
                        reward_dist,
                        pred_discount_dist,
                        _,
                        action_preds,
                    ) = self.world_model(
                        obs,
                        (high_level_actions, low_level_actions),
                        use_network_action=False,
                        batch_indices=batch_indices,
                        rt_idxs=rt_idxs,
                    )
                    obs = obs[
                        np.arange(batch_indices.shape[0]).reshape(-1, 1), batch_indices
                    ].reshape(-1, *self.image_shape)
                    rewards = rewards.reshape(-1, rewards.shape[-1])
                    terminals = terminals.reshape(-1, terminals.shape[-1])
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
                        {
                            k: v[
                                np.arange(batch_indices.shape[0]).reshape(-1, 1),
                                batch_indices,
                            ].reshape(-1, v.shape[-1])
                            for k, v in prior.items()
                        },
                        {
                            k: v[
                                np.arange(batch_indices.shape[0]).reshape(-1, 1),
                                batch_indices,
                            ].reshape(-1, v.shape[-1])
                            for k, v in post.items()
                        },
                        prior_dist,
                        post_dist,
                        pred_discount_dist,
                        obs,
                        rewards,
                        terminals,
                    )

                    batch_start = np.random.randint(
                        0,
                        low_level_actions.shape[1] - self.batch_length + 1,
                        size=(low_level_actions.shape[0]),
                    )
                    batch_indices = (
                        np.linspace(
                            batch_start,
                            batch_start + self.batch_length - 1,
                            self.batch_length - 1,
                            endpoint=False,
                        )
                        .astype(int)
                        .transpose(1, 0)
                    )
                    primitive_loss = self.criterion(
                        action_preds[
                            np.arange(batch_indices.shape[0]).reshape(-1, 1),
                            batch_indices,
                        ].reshape(-1, action_preds.shape[-1]),
                        low_level_actions[:, 1:][
                            np.arange(batch_indices.shape[0]).reshape(-1, 1),
                            batch_indices,
                        ].reshape(-1, action_preds.shape[-1]),
                    )
                    loss = world_model_loss + primitive_loss
                self.scaler.scale(loss * self.wm_loss_scale).backward()
                if itr == 0:
                    state = {k: v[:, rt_idxs].detach() for k, v in post.items()}
                else:
                    for k, v in state.items():
                        state[k] = torch.cat([v, post[k][:, rt_idxs].detach()])
            # by taking optimizer step outside the loop we are doing gradient accumulation
            self.update_network(
                self.world_model,
                self.world_model_optimizer,
                loss,
                self.world_model_gradient_clip,
            )

        """
        Actor Value Loss
        """
        vf_params = list(self.vf.parameters())
        target_vf_params = list(self.target_vf.parameters())
        log_keys = Counter()
        for _ in range(self.num_imagination_iterations):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    (imagined_features, imagined_actions, _) = self.imagine_ahead(state)
                    if self.world_model.reward_classifier:
                        imagined_reward = self.world_model.get_dist(
                            self.world_model.reward(imagined_features),
                            std=None,
                            normal=False,
                        ).mean
                        if self.binarize_rewards:
                            imagined_reward = (imagined_reward > 0.5).float()
                    else:
                        imagined_reward = self.world_model.reward(imagined_features)
                    if self.use_pred_discount:
                        discount = self.world_model.get_dist(
                            self.world_model.pred_discount(imagined_features),
                            std=None,
                            normal=False,
                        ).mean
                    else:
                        discount = self.discount * torch.ones_like(imagined_reward)
                    old_imagined_value = self.vf(imagined_features).detach()
                imagined_features_actions = (
                    imagined_features[:-1]
                    .reshape(-1, imagined_features.shape[-1])
                    .detach()
                )
                imagined_actions_actions = (
                    imagined_actions[:-1]
                    .reshape(-1, imagined_actions.shape[-1])
                    .detach()
                )
                imagined_actor_dist = self.actor(imagined_features_actions)
                imagined_log_probs = imagined_actor_dist.log_prob(
                    imagined_actions_actions
                ).detach()
            for _ in range(self.num_actor_value_updates):
                with torch.cuda.amp.autocast():
                    with FreezeParameters(vf_params + target_vf_params):
                        imagined_target_value = self.target_vf(imagined_features)
                        imagined_value = self.vf(imagined_features)
                    imagined_returns = lambda_return(
                        imagined_reward[:-1],
                        imagined_target_value[:-1],
                        discount[:-1],
                        bootstrap=imagined_target_value[-1],
                        lambda_=self.lam,
                    )
                    weights = torch.cumprod(
                        torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
                    ).detach()[:-1]

                    actor_loss = self.actor_loss(
                        imagined_returns,
                        imagined_value,
                        imagined_features,
                        imagined_actions,
                        weights,
                        imagined_log_probs,
                        self.actor,
                        log_keys,
                    )

                    with torch.no_grad():
                        imagined_features_values = imagined_features.detach()
                        target = imagined_returns.detach()
                        weights = weights.detach()

                    vf_loss = self.value_loss(
                        imagined_features_values,
                        weights,
                        target,
                        self.vf,
                        log_keys,
                        old_imagined_value,
                    )

                if self.use_actor_value_optimizer:
                    self.update_network(
                        [self.actor, self.vf],
                        self.actor_value_optimizer,
                        actor_loss + vf_loss,
                        self.actor_gradient_clip,
                    )
                else:
                    self.scaler.scale(actor_loss).backward()
                    self.update_network(
                        self.actor,
                        self.actor_optimizer,
                        actor_loss,
                        self.actor_gradient_clip,
                    )
                    self.scaler.scale(vf_loss).backward()
                    self.update_network(
                        self.vf,
                        self.vf_optimizer,
                        vf_loss,
                        self.value_gradient_clip,
                    )

        if self.num_imagination_iterations > 0:
            for key in log_keys:
                log_keys[key] /= (
                    self.num_actor_value_updates * self.num_imagination_iterations
                )
        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics["Primitive Model Loss"] = primitive_loss.item()
            eval_statistics["World Model Loss"] = world_model_loss.item()
            eval_statistics["Image Loss"] = image_pred_loss.item()
            eval_statistics["Reward Loss"] = reward_pred_loss.item()
            eval_statistics["Divergence Loss"] = div.item()
            eval_statistics["Transition Loss"] = transition_loss.item()
            eval_statistics["Entropy Loss"] = entropy_loss.item()
            eval_statistics["Pred Discount Loss"] = pred_discount_loss.item()
            if not self.world_model.discrete_latents:
                eval_statistics["Posterior State Std"] = post["std"].mean().item()
                eval_statistics["Prior State Std"] = prior["std"].mean().item()
            eval_statistics["Pred Discount Loss"] = pred_discount_loss.item()

            eval_statistics["Actor Loss"] = log_keys["actor_loss"]
            eval_statistics["Dynamics Backprop Loss"] = log_keys[
                "dynamics_backprop_loss"
            ]
            eval_statistics["Policy Gradient Loss"] = log_keys["policy_gradient_loss"]
            eval_statistics["Actor Entropy Loss"] = log_keys["actor_entropy_loss"]
            eval_statistics["Actor Entropy"] = -1 * log_keys["actor_entropy_loss"]
            eval_statistics["Actor Entropy Loss Scale"] = log_keys[
                "actor_entropy_loss_scale"
            ]
            eval_statistics["Actor Log Probs"] = log_keys["log_probs"]
            eval_statistics["Value Loss"] = log_keys["value_loss"]

            if self.num_imagination_iterations > 0:
                eval_statistics["Imagined Returns"] = imagined_returns.mean().item()
                eval_statistics["Imagined Rewards"] = imagined_reward.mean().item()
                eval_statistics["Imagined Values"] = log_keys["imagined_values_mean"]
            eval_statistics["Predicted Rewards"] = reward_dist.mean.mean().item()

        loss = DreamerLosses(
            actor_loss=actor_loss,
            world_model_loss=world_model_loss,
            vf_loss=vf_loss,
        )
        return loss, eval_statistics

    @torch.no_grad()
    def evaluate_helper(
        self,
        rewards,
        terminals,
        obs,
        target_obs,
        high_level_actions,
        low_level_actions,
        prefix,
        rt_idxs,
        use_network_action,
        forward_batch_method,
    ):
        with torch.cuda.amp.autocast():
            (
                post,
                prior,
                post_dist,
                prior_dist,
                image_dist,
                reward_dist,
                pred_discount_dist,
                _,
                action_preds,
            ) = self.world_model(
                obs,
                (high_level_actions, low_level_actions),
                use_network_action=use_network_action,
                batch_indices=rt_idxs,
                rt_idxs=rt_idxs,
                forward_batch_method=forward_batch_method,
            )
            obs = target_obs.reshape(-1, *self.image_shape)
            rewards = rewards.reshape(-1, rewards.shape[-1])
            terminals = terminals.reshape(-1, terminals.shape[-1])
            (_, _, image_pred_loss, _, _, _, _,) = self.world_model_loss(
                image_dist,
                reward_dist,
                {k: v[:, rt_idxs].reshape(-1, v.shape[-1]) for k, v in prior.items()},
                {k: v[:, rt_idxs].reshape(-1, v.shape[-1]) for k, v in post.items()},
                prior_dist,
                post_dist,
                pred_discount_dist,
                obs,
                rewards,
                terminals,
            )
            batch_start = np.random.randint(
                0,
                low_level_actions.shape[1] - self.batch_length + 1,
                size=(low_level_actions.shape[0]),
            )
            batch_indices = (
                np.linspace(
                    batch_start,
                    batch_start + self.batch_length - 1,
                    self.batch_length - 1,
                    endpoint=False,
                )
                .astype(int)
                .transpose(1, 0)
            )
            primitive_loss = self.criterion(
                action_preds[
                    np.arange(batch_indices.shape[0]).reshape(-1, 1),
                    batch_indices,
                ].reshape(-1, action_preds.shape[-1]),
                low_level_actions[:, 1:][
                    np.arange(batch_indices.shape[0]).reshape(-1, 1),
                    batch_indices,
                ].reshape(-1, action_preds.shape[-1]),
            )
        self.eval_statistics[prefix + "Image Loss"] = image_pred_loss.item()
        self.eval_statistics[
            prefix + "Reward Pred Error"
        ] = torch.nn.functional.mse_loss(rewards, reward_dist.mean).item()
        self.eval_statistics[
            prefix + "Pred Discount Error"
        ] = torch.nn.functional.mse_loss(pred_discount_dist.mean, terminals).item()
        self.eval_statistics[prefix + "Primitive Model Error"] = primitive_loss.item()

    @torch.no_grad()
    def evaluate(self, batch, buffer_data=True):
        if buffer_data:
            prefix = "replay/"
        else:
            prefix = "validation/"
        rewards = ptu.from_numpy(batch["rewards"])
        terminals = ptu.from_numpy(batch["terminals"])
        obs = ptu.from_numpy(batch["observations"])
        high_level_actions = ptu.from_numpy(batch["high_level_actions"])
        low_level_actions = ptu.from_numpy(batch["low_level_actions"])
        rt_idxs = np.arange(
            self.num_low_level_actions_per_primitive,
            obs.shape[1],
            self.num_low_level_actions_per_primitive,
        )
        # reset obs, effect of first primitive, second primitive, so on
        rt_idxs = np.concatenate([[0], rt_idxs])
        self.evaluate_helper(
            rewards,
            terminals,
            obs,
            obs[:, rt_idxs],
            high_level_actions,
            low_level_actions,
            prefix + "Full Obs ",
            rt_idxs,
            use_network_action=False,
            forward_batch_method="forward_batch",
        )

        self.evaluate_helper(
            rewards,
            terminals,
            obs,
            obs[:, rt_idxs],
            high_level_actions,
            low_level_actions,
            prefix + "Full Obs Primitive Model ",
            rt_idxs,
            use_network_action=True,
            forward_batch_method="forward_batch",
        )

        self.evaluate_helper(
            rewards,
            terminals,
            obs[:, rt_idxs],
            obs[:, rt_idxs],
            high_level_actions,
            low_level_actions,
            prefix + "RAPS Obs ",
            rt_idxs,
            use_network_action=False,
            forward_batch_method="forward_batch_raps_intermediate_obs",
        )

        self.evaluate_helper(
            rewards,
            terminals,
            obs[:, rt_idxs],
            obs[:, rt_idxs],
            high_level_actions,
            low_level_actions,
            prefix + "RAPS Obs Primitive Model ",
            rt_idxs,
            use_network_action=False,
            forward_batch_method="forward_batch_raps_intermediate_obs",
        )
