from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import kl_divergence as kld

import rlkit.torch.pytorch_util as ptu
from rlkit.core.loss import LossFunction, LossStatistics
from rlkit.torch.model_based.dreamer.actor_models import ConditionalActorModel
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
        target_vf,
        world_model,
        image_shape,
        imagination_horizon=None,
        discount=0.99,
        actor_lr=8e-5,
        vf_lr=8e-5,
        world_model_lr=3e-4,
        optimizer_class="torch_adam",
        use_amp=False,
        opt_level="O1",
        world_model_gradient_clip=100.0,
        actor_gradient_clip=100.0,
        value_gradient_clip=100.0,
        lam=0.95,
        free_nats=1.0,
        kl_loss_scale=0.0,
        pred_discount_loss_scale=10.0,
        image_loss_scale=1.0,
        reward_loss_scale=2.0,
        transition_loss_scale=0.8,
        forward_kl=False,
        policy_gradient_loss_scale=1.0,
        actor_entropy_loss_schedule="1e-4",
        adam_eps=1e-5,
        weight_decay=0.0,
        soft_target_tau=1,
        target_update_period=100,
        debug=False,
        initialize_amp=True,
        use_baseline=True,
        use_ppo_loss=False,
        ppo_clip_param=0.2,
        num_actor_value_updates=1,
        use_advantage_normalization=False,
        detach_rewards=False,
        num_imagination_iterations=1,
        use_clipped_value_loss=False,
        actor_value_lr=8e-5,
        use_actor_value_optimizer=False,
        state_loss_scale=0,
        train_decoder_on_second_output_only=False,
        use_pred_discount=True,
        reward_scale=1,
    ):
        super().__init__()

        torch.autograd.set_detect_anomaly(debug)

        torch.backends.cudnn.benchmark = True

        self.env = env
        self.use_pred_discount = use_pred_discount
        self.actor = actor.to(ptu.device)
        self.world_model = world_model.to(ptu.device)
        self.vf = vf.to(ptu.device)
        self.target_vf = target_vf.to(ptu.device)

        if optimizer_class == "torch_adam":
            self.optimizer_class = optim.Adam
        elif optimizer_class == "apex_adam" and APEX_AVAILABLE:
            self.optimizer_class = apex.optimizers.FusedAdam

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

        self.use_amp = use_amp and APEX_AVAILABLE
        if self.use_amp and initialize_amp:
            if self.use_actor_value_optimizer:
                models, optimizers = amp.initialize(
                    [
                        self.world_model.action_step_feature_extractor,
                        self.world_model.action_step_mlp,
                        self.world_model.obs_step_mlp,
                        self.world_model.conv_decoder,
                        self.world_model.conv_encoder,
                        self.world_model.pred_discount,
                        self.world_model.reward,
                        self.world_model.rnn,
                        self.actor,
                        self.vf,
                    ],
                    [
                        self.world_model_optimizer,
                        self.actor_value_optimizer,
                    ],
                    opt_level=opt_level,
                    num_losses=2,
                )
                (
                    self.world_model.action_step_feature_extractor,
                    self.world_model.action_step_mlp,
                    self.world_model.obs_step_mlp,
                    self.world_model.conv_decoder,
                    self.world_model.conv_encoder,
                    self.world_model.pred_discount,
                    self.world_model.reward,
                    self.world_model.rnn,
                    self.actor,
                    self.vf,
                ) = models
                (self.world_model_optimizer, self.actor_value_optimizer) = optimizers
            else:
                models, optimizers = amp.initialize(
                    [
                        self.world_model.action_step_feature_extractor,
                        self.world_model.action_step_mlp,
                        self.world_model.obs_step_mlp,
                        self.world_model.conv_decoder,
                        self.world_model.conv_encoder,
                        self.world_model.pred_discount,
                        self.world_model.reward,
                        self.world_model.rnn,
                        self.actor,
                        self.vf,
                    ],
                    [
                        self.world_model_optimizer,
                        self.actor_optimizer,
                        self.vf_optimizer,
                    ],
                    opt_level=opt_level,
                    num_losses=3,
                )
                (
                    self.world_model.action_step_feature_extractor,
                    self.world_model.action_step_mlp,
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
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()
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
        self.state_loss_scale = state_loss_scale
        self.train_decoder_on_second_output_only = train_decoder_on_second_output_only
        self.reward_scale = reward_scale

    def try_update_target_networks(self):
        if (
            self.target_vf
            and self._n_train_steps_total % self.target_update_period == 0
        ):
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def train_from_torch(self, batch):
        self.try_update_target_networks()
        _, stats = self.compute_loss(
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

    def imagine_ahead(self, state):

        new_state = {}
        for k, v in state.items():
            with torch.no_grad():
                if self.use_pred_discount:  # Last step could be terminal.
                    v = v[:, :-1]
                if k == "stoch" and self.world_model.discrete_latents:
                    new_state[k] = v.transpose(1, 0).reshape(
                        -1, v.shape[-2], v.shape[-1]
                    )
                else:
                    new_state[k] = v.transpose(1, 0).reshape(-1, v.shape[-1])
        feats = []
        actions = []
        for _ in range(self.imagination_horizon):
            feat = self.world_model.get_feat(new_state)
            action_dist = self.actor(feat.detach())
            action = action_dist.rsample()
            new_state = self.world_model.action_step(new_state, action)

            feats.append(feat.unsqueeze(0))
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
        if type(image_dist) == tuple:
            image_dist, state_dist = image_dist
            train_state_dist = True
        else:
            train_state_dist = False

        if self.train_decoder_on_second_output_only:
            image_dist = self.world_model.get_dist(
                mean=image_dist.base_dist.mean[:, 3:6, :, :],
                std=image_dist.base_dist.scale[:, 3:6, :, :],
                dims=3,
            )
            image_pred_loss = (
                -1
                * image_dist.log_prob(
                    self.world_model.preprocess(obs).reshape(-1, *self.image_shape)[
                        :, 3:6, :, :
                    ]
                ).mean()  # todo: need to actually choose the params of the image dist to be [3:6]
            )
        else:
            image_pred_loss = (
                -1
                * image_dist.log_prob(
                    self.world_model.preprocess(obs).reshape(-1, *self.image_shape)
                ).mean()
            )
        if self.detach_rewards:
            reward_pred_loss = -1 * reward_dist.log_prob(rewards.detach()).mean()
        else:
            reward_pred_loss = -1 * reward_dist.log_prob(rewards).mean()
        pred_discount_target = self.discount * (1 - terminals.float())
        pred_discount_loss = (
            -1 * pred_discount_dist.log_prob(pred_discount_target).mean()
        )

        if self.world_model.discrete_latents:
            post_detached_dist = self.world_model.get_detached_dist(
                post["logits"],
                post["logits"],
                latent=True,
            )
            prior_detached_dist = self.world_model.get_detached_dist(
                prior["logits"],
                prior["logits"],
                latent=True,
            )
        else:
            post_detached_dist = self.world_model.get_detached_dist(
                post["mean"],
                post["std"],
                latent=True,
            )
            prior_detached_dist = self.world_model.get_detached_dist(
                prior["mean"],
                prior["std"],
                latent=True,
            )
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
        if train_state_dist:
            state = obs[:, np.prod(self.image_shape) :]
            state_dist_loss = -1 * state_dist.log_prob(state).mean()
            world_model_loss += self.state_loss_scale * state_dist_loss
            return (
                world_model_loss,
                div,
                (image_pred_loss, state_dist_loss),
                reward_pred_loss,
                transition_loss,
                entropy_loss,
                pred_discount_loss,
            )
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
        imag_returns,
        value,
        imag_feat,
        imag_actions,
        weights,
        old_imag_log_probs,
        actor,
    ):
        assert len(imag_returns.shape) == 3, imag_returns.shape
        assert len(value.shape) == 3, value.shape
        assert len(imag_feat.shape) == 3, imag_feat.shape
        assert len(imag_actions.shape) == 3, imag_actions.shape
        assert len(weights.shape) == 3 and weights.shape[-1] == 1, weights.shape
        if actor.use_tanh_normal:
            assert imag_actions.max() <= 1.0 and imag_actions.min() >= -1.0
        if self.use_baseline:
            advantages = imag_returns - value[:-1]
        else:
            advantages = imag_returns
        if self.use_advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        advantages = advantages.reshape(-1).detach()

        imag_feat_a = imag_feat[:-1].reshape(-1, imag_feat.shape[-1]).detach()
        imag_actions = imag_actions[:-1].reshape(-1, imag_actions.shape[-1]).detach()
        assert imag_actions.shape[0] == imag_feat_a.shape[0]

        imag_actor_dist = actor(imag_feat_a)
        imag_log_probs = imag_actor_dist.log_prob(imag_actions)
        assert old_imag_log_probs.shape == imag_log_probs.shape
        if self.use_ppo_loss:
            ratio = torch.exp(imag_log_probs - old_imag_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.ppo_clip_param, 1.0 + self.ppo_clip_param)
                * advantages
            )
            policy_gradient_loss = -torch.min(surr1, surr2)
        else:
            policy_gradient_loss = -1 * imag_log_probs * advantages
        actor_entropy_loss = -1 * imag_actor_dist.entropy()

        dynamics_backprop_loss = -(imag_returns)
        dynamics_backprop_loss = dynamics_backprop_loss.reshape(-1)
        weights = weights.reshape(-1)
        assert (
            dynamics_backprop_loss.shape
            == policy_gradient_loss.shape
            == actor_entropy_loss.shape
            == weights.shape
        )
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
        return (
            actor_loss,
            dynamics_backprop_loss.mean(),
            policy_gradient_loss.mean(),
            actor_entropy_loss.mean(),
            actor_entropy_loss_scale,
            imag_log_probs.mean(),
        )

    def value_loss(
        self,
        imag_feat_v,
        weights,
        imag_returns,
        vf,
        old_imag_value=None,
    ):
        assert len(imag_feat_v.shape) == 3, imag_feat_v.shape
        assert len(weights.shape) == 3 and weights.shape[-1] == 1, weights.shape
        assert len(imag_returns.shape) == 3, imag_returns.shape

        values = vf(imag_feat_v)[:-1]
        value_dist = self.world_model.get_dist(values, 1)
        if self.use_clipped_value_loss:
            value_pred_clipped = old_imag_value[:-1] + (
                values - old_imag_value[:-1]
            ).clamp(-self.ppo_clip_param, self.ppo_clip_param)
            vf_losses = value_dist.log_prob(imag_returns)
            vf_clipped_dist = self.world_model.get_dist(value_pred_clipped, 1)
            vf_losses_clipped = vf_clipped_dist.log_prob(imag_returns)
            weights = weights.squeeze(-1)
            vf_loss = (torch.max(-vf_losses, -vf_losses_clipped) * weights).mean()
        else:
            log_probs = value_dist.log_prob(imag_returns)
            weights = weights.squeeze(-1)
            assert len(log_probs.shape) == 2, log_probs.shape
            assert weights.shape == log_probs.shape
            vf_loss = -(weights * log_probs).mean()
        return vf_loss, value_dist.mean.mean()

    def update_network(self, network, optimizer, loss, loss_id, gradient_clip):
        if type(network) == list:
            parameters = []
            for net in network:
                zero_grad(net)
                parameters.extend(list(net.parameters()))
        else:
            zero_grad(network)
            parameters = list(network.parameters())
        if self.use_amp:
            with amp.scale_loss(loss, optimizer, loss_id=loss_id) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if gradient_clip > 0:
            if not self.use_amp:
                torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, norm_type=2)
            else:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer),
                    gradient_clip,
                    norm_type=2,
                )
        optimizer.step()

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
        **kwargs,
    ) -> Tuple[DreamerLosses, LossStatistics]:
        rewards = batch["rewards"] * self.reward_scale
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
            _,
        ) = self.world_model(obs, actions)
        # stack obs, rewards and terminals along path dimension
        obs = obs.transpose(1, 0).reshape(-1, np.prod(self.image_shape))
        rewards = rewards.transpose(1, 0).reshape(-1, rewards.shape[-1])
        terminals = terminals.transpose(1, 0).reshape(-1, terminals.shape[-1])

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
            self.world_model,
            self.world_model_optimizer,
            world_model_loss,
            0,
            self.world_model_gradient_clip,
        )

        """
        Actor Value Loss
        """
        world_model_params = list(self.world_model.parameters())
        vf_params = list(self.vf.parameters())
        target_vf_params = list(self.target_vf.parameters())
        pred_discount_params = list(self.world_model.pred_discount.parameters())
        (
            actor_loss,
            dynamics_backprop_loss,
            policy_gradient_loss,
            actor_entropy_loss,
            actor_entropy_loss_scale,
            log_probs,
            vf_loss,
            imag_values_mean,
        ) = (0, 0, 0, 0, 0, 0, 0, 0)
        for _ in range(self.num_imagination_iterations):

            with FreezeParameters(world_model_params + pred_discount_params):
                (
                    imag_feat,
                    imag_actions,
                ) = self.imagine_ahead(post)
                imag_reward = self.world_model.reward(imag_feat)
                if self.use_pred_discount:
                    discount = self.world_model.get_dist(
                        self.world_model.pred_discount(imag_feat),
                        std=None,
                        normal=False,
                    ).mean
                else:
                    discount = self.discount * torch.ones_like(imag_reward)
            with FreezeParameters(vf_params):
                old_imag_value = self.vf(imag_feat).detach()
            imag_feat_a = imag_feat[:-1].reshape(-1, imag_feat.shape[-1]).detach()
            imag_actions_a = (
                imag_actions[:-1].reshape(-1, imag_actions.shape[-1]).detach()
            )
            imag_actor_dist = self.actor(imag_feat_a)
            imag_log_probs = imag_actor_dist.log_prob(imag_actions_a).detach()
            for _ in range(self.num_actor_value_updates):
                with FreezeParameters(vf_params + target_vf_params):
                    imag_target_value = self.target_vf(imag_feat)
                    imag_value = self.vf(imag_feat)
                imag_returns = lambda_return(
                    imag_reward[:-1],
                    imag_target_value[:-1],
                    discount[:-1],
                    bootstrap=imag_target_value[-1],
                    lambda_=self.lam,
                )
                weights = torch.cumprod(
                    torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
                ).detach()[:-1]

                (
                    actor_loss_,
                    dynamics_backprop_loss_,
                    policy_gradient_loss_,
                    actor_entropy_loss_,
                    actor_entropy_loss_scale_,
                    log_probs_,
                ) = self.actor_loss(
                    imag_returns,
                    imag_value,
                    imag_feat,
                    imag_actions,
                    weights,
                    imag_log_probs,
                    self.actor,
                )

                with torch.no_grad():
                    imag_feat_v = imag_feat.detach()
                    target = imag_returns.detach()
                    weights = weights.detach()

                vf_loss_, imag_values_mean_ = self.value_loss(
                    imag_feat_v, weights, target, self.vf, old_imag_value
                )

                if self.use_actor_value_optimizer:
                    self.update_network(
                        [self.actor, self.vf],
                        self.actor_value_optimizer,
                        actor_loss_ + vf_loss_,
                        1,
                        self.actor_gradient_clip,
                    )
                else:
                    self.update_network(
                        self.actor,
                        self.actor_optimizer,
                        actor_loss_,
                        1,
                        self.actor_gradient_clip,
                    )

                    self.update_network(
                        self.vf,
                        self.vf_optimizer,
                        vf_loss_,
                        2,
                        self.value_gradient_clip,
                    )
                actor_loss += actor_loss_.item()
                dynamics_backprop_loss += dynamics_backprop_loss_.item()
                policy_gradient_loss += policy_gradient_loss_.item()
                actor_entropy_loss += actor_entropy_loss_.item()
                actor_entropy_loss_scale += actor_entropy_loss_scale_
                log_probs += log_probs_.item()
                vf_loss += vf_loss_.item()
                imag_values_mean += imag_values_mean_.item()

        if self.num_imagination_iterations > 0:
            actor_loss /= self.num_actor_value_updates * self.num_imagination_iterations
            dynamics_backprop_loss /= (
                self.num_actor_value_updates * self.num_imagination_iterations
            )
            policy_gradient_loss /= (
                self.num_actor_value_updates * self.num_imagination_iterations
            )
            actor_entropy_loss /= (
                self.num_actor_value_updates * self.num_imagination_iterations
            )
            actor_entropy_loss_scale /= (
                self.num_actor_value_updates * self.num_imagination_iterations
            )
            log_probs /= self.num_actor_value_updates * self.num_imagination_iterations
            vf_loss /= self.num_actor_value_updates * self.num_imagination_iterations
            imag_values_mean /= (
                self.num_actor_value_updates * self.num_imagination_iterations
            )

        if type(image_pred_loss) == tuple:
            image_pred_loss, state_pred_loss = image_pred_loss
            log_state_pred_loss = True
        else:
            log_state_pred_loss = False

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics["World Model Loss"] = world_model_loss.item()
            eval_statistics["Image Loss"] = image_pred_loss.item()
            if log_state_pred_loss:
                eval_statistics["State Prediction Loss"] = state_pred_loss.item()
            eval_statistics["Reward Loss"] = reward_pred_loss.item()
            eval_statistics["Divergence Loss"] = div.item()
            eval_statistics["Transition Loss"] = transition_loss.item()
            eval_statistics["Entropy Loss"] = entropy_loss.item()
            eval_statistics["Pred Discount Loss"] = pred_discount_loss.item()
            if not self.world_model.discrete_latents:
                eval_statistics["Posterior State Std"] = post["std"].mean().item()
                eval_statistics["Prior State Std"] = prior["std"].mean().item()
            eval_statistics["Pred Discount Loss"] = pred_discount_loss.item()

            eval_statistics["Actor Loss"] = actor_loss
            eval_statistics["Dynamics Backprop Loss"] = dynamics_backprop_loss
            eval_statistics["Policy Gradient Loss"] = policy_gradient_loss
            eval_statistics["Actor Entropy Loss"] = actor_entropy_loss
            eval_statistics["Actor Entropy Loss Scale"] = actor_entropy_loss_scale
            eval_statistics["Actor Log Probs"] = log_probs
            eval_statistics["Value Loss"] = vf_loss

            if self.num_imagination_iterations > 0:
                eval_statistics["Imagined Returns"] = imag_returns.mean().item()
                eval_statistics["Imagined Rewards"] = imag_reward.mean().item()
            eval_statistics["Imagined Values"] = imag_values_mean
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
        return dict(
            actor=self.actor,
            world_model=self.world_model,
            vf=self.vf,
            target_vf=self.target_vf,
        )

    def pretrain_actor_vf(self, num_imagination_iterations):
        """
        Actor Value Loss
        """
        world_model_params = list(self.world_model.parameters())
        vf_params = list(self.vf.parameters())
        target_vf_params = list(self.target_vf.parameters())
        pred_discount_params = list(self.world_model.pred_discount.parameters())
        null_state = self.world_model.initial(2500)
        null_acts = ptu.zeros((2500, self.env.action_space.low.size))
        reset_obs = ptu.from_numpy(np.concatenate([self.env.reset()])).repeat((2500, 1))
        embed = self.world_model.encode(reset_obs)
        post, _ = self.world_model.obs_step(null_state, null_acts, embed)
        post_ = {}
        for k, v in post.items():
            post_[k] = v.reshape(50, 50, -1)
        post = post_
        (
            actor_loss,
            dynamics_backprop_loss,
            policy_gradient_loss,
            actor_entropy_loss,
            actor_entropy_loss_scale,
            log_probs,
            vf_loss,
            imag_values_mean,
        ) = (0, 0, 0, 0, 0, 0, 0, 0)
        for _ in range(num_imagination_iterations):

            with FreezeParameters(world_model_params + pred_discount_params):
                (
                    imag_feat,
                    imag_actions,
                ) = self.imagine_ahead(post)
                imag_reward = self.world_model.reward(imag_feat)
                if self.use_pred_discount:
                    discount = self.world_model.get_dist(
                        self.world_model.pred_discount(imag_feat),
                        std=None,
                        normal=False,
                    ).mean
                else:
                    discount = self.discount * torch.ones_like(imag_reward)
            imag_feat_a = imag_feat[:-1].reshape(-1, imag_feat.shape[-1]).detach()
            imag_actions_a = (
                imag_actions[:-1].reshape(-1, imag_actions.shape[-1]).detach()
            )
            imag_actor_dist = self.actor(imag_feat_a)
            imag_log_probs = imag_actor_dist.log_prob(imag_actions_a).detach()
            with FreezeParameters(vf_params):
                old_imag_value = self.vf(imag_feat).detach()
            for _ in range(self.num_actor_value_updates):
                with FreezeParameters(vf_params + target_vf_params):
                    imag_target_value = self.target_vf(imag_feat)
                    imag_value = self.vf(imag_feat)
                imag_returns = lambda_return(
                    imag_reward[:-1],
                    imag_target_value[:-1],
                    discount[:-1],
                    bootstrap=imag_target_value[-1],
                    lambda_=self.lam,
                )
                weights = torch.cumprod(
                    torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
                ).detach()[:-1]

                (
                    actor_loss_,
                    dynamics_backprop_loss_,
                    policy_gradient_loss_,
                    actor_entropy_loss_,
                    actor_entropy_loss_scale_,
                    log_probs_,
                ) = self.actor_loss(
                    imag_returns,
                    imag_value,
                    imag_feat,
                    imag_actions,
                    weights,
                    imag_log_probs,
                    self.actor,
                )

                with torch.no_grad():
                    imag_feat_v = imag_feat.detach()
                    target = imag_returns.detach()
                    weights = weights.detach()

                vf_loss_, imag_values_mean_ = self.value_loss(
                    imag_feat_v, weights, target, self.vf, old_imag_value
                )

                if self.use_actor_value_optimizer:
                    self.update_network(
                        [self.actor, self.vf],
                        self.actor_value_optimizer,
                        actor_loss_ + vf_loss_,
                        1,
                        self.actor_gradient_clip,
                    )
                else:
                    self.update_network(
                        self.actor,
                        self.actor_optimizer,
                        actor_loss_,
                        1,
                        self.actor_gradient_clip,
                    )

                    self.update_network(
                        self.vf,
                        self.vf_optimizer,
                        vf_loss_,
                        2,
                        self.value_gradient_clip,
                    )
                actor_loss += actor_loss_.item()
                dynamics_backprop_loss += dynamics_backprop_loss_.item()
                policy_gradient_loss += policy_gradient_loss_.item()
                actor_entropy_loss += actor_entropy_loss_.item()
                actor_entropy_loss_scale += actor_entropy_loss_scale_
                log_probs += log_probs_.item()
                vf_loss += vf_loss_.item()
                imag_values_mean += imag_values_mean_.item()

        if self.num_imagination_iterations > 0:
            actor_loss /= self.num_actor_value_updates * self.num_imagination_iterations
            dynamics_backprop_loss /= (
                self.num_actor_value_updates * self.num_imagination_iterations
            )
            policy_gradient_loss /= (
                self.num_actor_value_updates * self.num_imagination_iterations
            )
            actor_entropy_loss /= (
                self.num_actor_value_updates * self.num_imagination_iterations
            )
            actor_entropy_loss_scale /= (
                self.num_actor_value_updates * self.num_imagination_iterations
            )
            log_probs /= self.num_actor_value_updates * self.num_imagination_iterations
            vf_loss /= self.num_actor_value_updates * self.num_imagination_iterations
            imag_values_mean /= (
                self.num_actor_value_updates * self.num_imagination_iterations
            )
