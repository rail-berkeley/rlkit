import os
import os.path as osp
import pickle
from collections import Counter, OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.distributions import kl_divergence as kld

import rlkit.torch.pytorch_util as ptu
from rlkit.core.loss import LossFunction, LossStatistics
from rlkit.torch.model_based.dreamer.utils import *
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class DreamerV2Trainer(TorchTrainer, LossFunction):
    def __init__(
        self,
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
        binarize_rewards=False,
    ):
        super().__init__()

        torch.backends.cudnn.benchmark = True

        self.scaler = torch.cuda.amp.GradScaler()
        self.use_pred_discount = use_pred_discount
        self.actor = actor.to(ptu.device)
        self.world_model = world_model.to(ptu.device)
        self.vf = vf.to(ptu.device)
        self.target_vf = target_vf.to(ptu.device)

        optimizer_class = optim.Adam

        self.actor_lr = actor_lr
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.vf_lr = vf_lr
        self.world_model_lr = world_model_lr

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
        self.use_actor_value_optimizer = use_actor_value_optimizer
        self.actor_value_optimizer = optimizer_class(
            list(self.actor.parameters()) + list(self.vf.parameters()),
            lr=actor_value_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )

        self.discount = discount
        self.lam = lam
        self.imagination_horizon = imagination_horizon
        self.free_nats = ptu.tensor(free_nats)
        self.kl_loss_scale = kl_loss_scale
        self.pred_discount_loss_scale = pred_discount_loss_scale
        self.image_loss_scale = image_loss_scale
        self.reward_loss_scale = reward_loss_scale
        self.transition_loss_scale = transition_loss_scale
        self.policy_gradient_loss_scale = policy_gradient_loss_scale
        self.actor_entropy_loss_schedule = actor_entropy_loss_schedule
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
        self.use_dynamics_backprop = self.policy_gradient_loss_scale < 1.0
        self.binarize_rewards = binarize_rewards

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def train_from_torch(self, batch):
        self.try_update_target_networks()
        stats = self.train_networks(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )

        self._n_train_steps_total += 1
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            self._need_to_update_eval_statistics = False

    def compute_world_model_loss_and_state(self, batch, log_keys):
        """
        :param batch: Dict
            rewards: (batch_size, path_length+1, 1)
            terminals: (batch_size, path_length+1, 1)
            observations: (batch_size, path_length+1, obs_dim)
            actions: (batch_size, path_length+1, act_dim)
        :log_keys: Counter

        :return state: Dict
            stoch: torch.Tensor (batch_size, path_length+1, stoch_size)
            deter: torch.Tensor (batch_size, path_length+1, deter_size)
            mean: torch.Tensor (batch_size, path_length+1, stoch_size)
            std: torch.Tensor (batch_size, path_length+1, stoch_size)
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
            obs = self.world_model.get_image_from_obs(obs).reshape(
                -1, *self.image_shape
            )

            rewards = rewards.reshape(-1, rewards.shape[-1])
            terminals = terminals.reshape(-1, terminals.shape[-1])
            world_model_loss = self.world_model_loss(
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
                log_keys,
            )
        state = post
        self.scaler.scale(world_model_loss).backward()
        log_keys["Reward in Batch"] = rewards.sum().item()
        return state

    @torch.cuda.amp.autocast()
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
        log_keys,
    ):
        """
        :param image_dist: Normal(mean, 1)
            mu: torch.Tensor (batch_size*(path_length+1), image_shape)
        :param reward_dist: Normal(mean, 1)
            mu: torch.Tensor (batch_size*(path_length+1), 1)
        :param prior: Dict
            stoch: torch.Tensor (batch_size, path_length+1, stoch_size)
            mean: torch.Tensor (batch_size, path_length+1, stoch_size)
            std: torch.Tensor (batch_size, path_length+1, stoch_size)
            deter: torch.Tensor (batch_size, path_length+1, deter_size)
        :param post: Dict
            stoch: torch.Tensor (batch_size, path_length+1, stoch_size)
            mean: torch.Tensor (batch_size, path_length+1, stoch_size)
            std: torch.Tensor (batch_size, path_length+1, stoch_size)
            deter: torch.Tensor (batch_size, path_length+1, deter_size)
        :param prior_dist: Normal(mean, std)
            mean: torch.Tensor (batch_size*(path_length+1), stoch_size)
            std: torch.Tensor (batch_size*(path_length+1), stoch_size)
        :param post_dist: Normal(mean, std)
            mean: torch.Tensor (batch_size*(path_length+1), stoch_size)
            std: torch.Tensor (batch_size*(path_length+1), stoch_size)
        :param pred_discount_dist: Bernoulli(logits, 1)
            logits: torch.Tensor (batch_size*(path_length+1), 1)
        :param obs: torch.Tensor (batch_size*(path_length+1), image_shape)
        :param rewards: torch.Tensor (batch_size*(path_length+1), 1)
        :param terminals: torch.Tensor (batch_size*(path_length+1), 1)

        :return world_model_loss:
        """
        assert (
            image_dist.mean.shape[0]
            == prior_dist.mean.shape[0]
            == post_dist.mean.shape[0]
            == obs.shape[0]
        ), f"Batch dimension should be the same. Got {image_dist.mean.shape, prior_dist.mean.shape, post_dist.mean.shape, obs.shape}."
        assert (
            reward_dist.mean.shape[0]
            == pred_discount_dist.mean.shape[0]
            == rewards.shape[0]
            == terminals.shape[0]
        ), f"Batch dimension should be the same. Got {reward_dist.mean.shape, pred_discount_dist.mean.shape, rewards.shape, terminals.shape}."
        assert obs.max() > 1, f"Obs should not be preprocessed yet. Got {obs.max()}."
        reshaped_post = {}
        reshaped_prior = {}
        for k, v in post.items():
            assert (
                len(v.shape) == 3
            ), f"{k} should be of shape (batch_size, path_length+1, ...). Got {v.shape}."
            reshaped_post[k] = v.reshape(-1, v.shape[-1])

        for k, v in prior.items():
            assert (
                len(v.shape) == 3
            ), f"{k} should be of shape (batch_size, path_length+1, ...). Got {v.shape}."
            reshaped_prior[k] = v.reshape(-1, v.shape[-1])

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

        post_detached_dist = self.world_model.get_detached_dist(reshaped_post)
        prior_detached_dist = self.world_model.get_detached_dist(reshaped_prior)
        if self.forward_kl:
            div = kld(post_dist, prior_dist).mean()
            div = torch.max(div, self.free_nats)
            prior_kld = kld(post_detached_dist, prior_dist).mean()
            post_kld = kld(post_dist, prior_detached_dist).mean()
        else:
            div = kld(prior_dist, post_dist).mean()
            div = torch.max(div, self.free_nats)
            prior_kld = kld(prior_dist, post_detached_dist).mean()
            post_kld = kld(prior_detached_dist, post_dist).mean()
        transition_loss = torch.max(prior_kld, self.free_nats)
        entropy_loss = torch.max(post_kld, self.free_nats)
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
        log_keys["World Model Loss"] += world_model_loss.item()
        log_keys["World Model Image Prediction Loss"] += image_pred_loss.item()
        log_keys["World Model Reward Prediction Loss"] += reward_pred_loss.item()
        log_keys["World Model Transition Loss"] += transition_loss.item()
        log_keys["World Model Entropy Loss"] += entropy_loss.item()
        log_keys["World Model Pred Discount Loss"] += pred_discount_loss.item()
        log_keys["World Model Divergence Loss"] += div.item()
        assert (
            world_model_loss.requires_grad == True
        ), "World model loss should require gradients."
        return world_model_loss

    def imagine_ahead(self, state, actor=None):
        """
        Imagine forward using actor and world model (no observations provided).

        :param state Dict
            stoch: torch.Tensor (batch_size, path_length+1, stoch_size)
            mean: torch.Tensor (batch_size, path_length+1, stoch_size)
            std: torch.Tensor (batch_size, path_length+1, stoch_size)
            deter: torch.Tensor (batch_size, path_length+1, deter_size)
        :param actor: torch.nn.Module

        :return imagined_features: torch.Tensor(imagination_horizon, batch_size*path_length, stoch_size + deter_size)
        :return imagined_actions: torch.Tensor(imagination_horizon, batch_size*path_length, action_size)
        """
        for k, v in state.items():
            assert (
                len(v.shape) == 3
            ), f"{k} should be of shape (batch_size, path_length+1, size). Got {v.shape}."
        torch.set_grad_enabled(self.use_dynamics_backprop)
        if actor is None:
            actor = self.actor
        new_state = {}
        for key, value in state.items():
            if self.use_pred_discount:
                value = value[:, :-1]
            new_state[key] = value.reshape(-1, value.shape[-1]).detach()
        imagined_features = []
        imagined_actions = []
        new_states = dict(mean=[], std=[], stoch=[], deter=[])
        for _ in range(self.imagination_horizon):
            features = self.world_model.get_features(new_state)
            for key in new_states.keys():
                new_states[key].append(new_state[key].unsqueeze(0))
            action_dist = actor(features.detach())
            action = action_dist.rsample()
            new_state = self.world_model.action_step(new_state, action)

            imagined_features.append(features.unsqueeze(0))
            imagined_actions.append(action.unsqueeze(0))
        imagined_features = torch.cat(imagined_features)
        imagined_actions = torch.cat(imagined_actions)
        for key in new_states.keys():
            new_states[key] = torch.cat(new_states[key])
        torch.set_grad_enabled(True)
        assert (
            imagined_features.shape[:2] == imagined_actions.shape[:2]
        ), "Imagined features and actions should have the same first two dimensions."
        assert (
            imagined_features.shape[0]
            == imagined_actions.shape[0]
            == self.imagination_horizon
        ), "Imagined dim 0 should be horizon."
        return imagined_features, imagined_actions, new_states

    @torch.cuda.amp.autocast()
    def collect_imagination_data(self, state):
        """
        Run imagination for a single batch of data. Compute rewards, discounts and values.
        :param state Dict
            stoch: torch.Tensor (batch_size, path_length+1, stoch_size)
            mean: torch.Tensor (batch_size, path_length+1, stoch_size)
            std: torch.Tensor (batch_size, path_length+1, stoch_size)
            deter: torch.Tensor (batch_size, path_length+1, deter_size)

        :return imagined_features: torch.Tensor(imagination_horizon, batch_size*path_length, stoch_size + deter_size)
        :return imagined_actions: torch.Tensor(imagination_horizon, batch_size*path_length, action_size)
        :return imagined_discount: torch.Tensor (imagine_horizon, batch_size*path_length, 1)
        :return imagined_reward: torch.Tensor (imagination_horizon, batch_size*path_length, 1)
        :return imagined_value: torch.Tensor (imagination_horizon, batch_size*path_length, 1)
        """
        torch.set_grad_enabled(self.use_dynamics_backprop)
        # Standard dreamer (with dynamics backprop) breaks if you remove this.
        # TODO (mdalal): write a test to figure out why and avoid this issue
        with FreezeParameters(list(self.world_model.parameters())):
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
                imagined_discount = self.world_model.get_dist(
                    self.world_model.pred_discount(imagined_features),
                    std=None,
                    normal=False,
                ).mean
            else:
                imagined_discount = self.discount * torch.ones_like(imagined_reward)
        imagined_value = self.vf(imagined_features).detach()
        torch.set_grad_enabled(True)
        assert (
            imagined_features.shape[:2]
            == imagined_actions.shape[:2]
            == imagined_discount.shape[:2]
            == imagined_reward.shape[:2]
            == imagined_value.shape[:2]
        ), "First two dimensions of all return values should be equal."
        return (
            imagined_features,
            imagined_actions,
            imagined_discount,
            imagined_reward,
            imagined_value,
        )

    @torch.cuda.amp.autocast()
    def compute_values_for_training_actor_and_value(
        self, imagined_features, imagined_reward, imagined_discount
    ):
        """
        Compute imagined returns and associated quantities.

        :param imagined_features: torch.Tensor (imagination_horizon, batch_size*path_length, stoch_size + deter_size)
        :param imagined_reward: torch.Tensor (imagination_horizon, batch_size*path_length, 1)
        :param imagined_discount: torch.Tensor (imagination_horizon, batch_size*path_length, 1)

        :return imagined_value: torch.Tensor (imagination_horizon, batch_size*path_length, 1)
        :return imagined_return: torch.Tensor (imagination_horizon-1, batch_size*path_length, 1)
        :return weights: torch.Tensor (imagination_horizon-1, batch_size*path_length, 1)
        :return imagined_features: torch.Tensor(imagination_horizon, batch_size*path_length, stoch_size + deter_size)
        """
        torch.set_grad_enabled(self.use_dynamics_backprop)
        with FreezeParameters(
            list(self.vf.parameters()) + list(self.target_vf.parameters())
        ):
            imagined_target_value = self.target_vf(imagined_features)
            imagined_value = self.vf(imagined_features)
        imagined_return = lambda_return(
            imagined_reward[:-1],
            imagined_target_value[:-1],
            imagined_discount[:-1],
            bootstrap=imagined_target_value[-1],
            lambda_=self.lam,
        )
        weights = compute_weights_from_discount(imagined_discount)
        torch.set_grad_enabled(True)
        assert (
            imagined_return.shape == weights.shape
        ), "Imagined return and weights should have same shape."
        assert (
            imagined_value.shape[:2] == imagined_features.shape[:2]
        ), "Imagined value and features should have the same first two dimensions."
        return (
            imagined_value,
            imagined_return,
            weights,
            imagined_features,
        )

    @torch.cuda.amp.autocast()
    def actor_loss(
        self,
        imagined_returns,
        value,
        imagined_features,
        imagined_actions,
        weights,
        old_imagined_log_probs,
        actor,
        log_keys: Counter,
        prefix: str = "",
    ):
        """
        :param imagined_returns: torch.Tensor (imagination_horizon-1, batch_size*path_length, 1)
        :param value: torch.Tensor (imagination_horizon-1, batch_size*path_length, 1)
        :param imagined_features: torch.Tensor(imagination_horizon, batch_size*path_length, stoch_size + deter_size)
        :param imagined_actions: torch.Tensor (imagination_horizon, batch_size*path_length, action_size)
        :param weights: torch.Tensor (imagination_horizon-1, batch_size*path_length, 1)
        :param old_imagined_log_probs: torch.Tensor (imagination_horizon-1, batch_size*path_length, 1)
        :param actor: torch.nn.Module
        :param log_keys:
        :param prefix: str

        :return actor_loss:
        """
        if self.use_baseline:
            advantages = imagined_returns - value[:-1]
        else:
            advantages = imagined_returns
        if self.use_advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        advantages = advantages.reshape(-1).detach()

        imagined_actor_dist = actor(
            imagined_features[:-1].reshape(-1, imagined_features.shape[-1]).detach()
        )
        imagined_log_probs = imagined_actor_dist.log_prob(
            imagined_actions[:-1].reshape(-1, imagined_actions.shape[-1]).detach()
        )
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
        actor_loss = (
            (
                dynamics_backprop_loss_scale * dynamics_backprop_loss
                + self.policy_gradient_loss_scale * policy_gradient_loss
                + actor_entropy_loss_scale * actor_entropy_loss
            )
            * weights
        ).mean()

        assert actor_loss.requires_grad is True, "Actor loss should require gradients."
        if self.use_dynamics_backprop:
            assert (
                dynamics_backprop_loss.requires_grad is True
            ), "Dynamics backprop loss should require gradients if True."

        log_keys[prefix + "Actor Loss"] += actor_loss.item()
        log_keys[
            prefix + "Actor Dynamics Backprop Loss"
        ] += dynamics_backprop_loss.mean().item()
        log_keys[prefix + "Actor Entropy Loss"] += actor_entropy_loss.mean().item()
        log_keys[prefix + "Actor Entropy"] += -1 * actor_entropy_loss.mean().item()
        log_keys[prefix + "Actor Entropy Loss Scale"] += actor_entropy_loss_scale
        log_keys[
            prefix + "Actor Policy Gradient Loss"
        ] += policy_gradient_loss.mean().item()
        log_keys[prefix + "Actor Log Probs"] += imagined_log_probs.mean().item()

        return actor_loss

    @torch.cuda.amp.autocast()
    def value_loss(
        self,
        imagined_features_v,
        weights,
        imagined_returns,
        vf,
        log_keys: Counter,
        old_imagined_value=None,
        prefix: str = "",
    ):
        """
        :param imagined_features_v: torch.Tensor (imagination_horizon, batch_size*path_length, stoch_size + deter_size)
        :param weights: torch.Tensor (imagination_horizon-1, batch_size*path_length, 1)
        :param imagined_returns: torch.Tensor (imagination_horizon-1, batch_size*path_length, 1)
        :param vf: torch.nn.Module
        :param log_keys: Counter
        :param old_imagined_value: torch.Tensor (imagination_horizon-1, batch_size*path_length, 1)
        :param prefix:str

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
        log_keys[prefix + "Value Loss"] += vf_loss.item()
        log_keys[prefix + "Imagined Value Mean"] += value_dist.mean.mean().item()
        assert vf_loss.requires_grad, "Value loss should require gradients."
        return vf_loss

    def train_networks(
        self,
        batch,
        skip_statistics=False,
    ) -> LossStatistics:
        """
        :param batch: dict
            rewards: (batch_size, path_length+1, 1)
            terminals: (batch_size, path_length+1, 1)
            observations: (batch_size, path_length+1, obs_dim)
            actions: (batch_size, path_length+1, act_dim)
        :param skip_statistics: bool

        :return eval_statistics: LossStatistics
        """

        """
        World Model Loss
        """
        log_keys = Counter()

        state = self.compute_world_model_loss_and_state(batch, log_keys)
        update_network(
            self.world_model,
            self.world_model_optimizer,
            self.world_model_gradient_clip,
            self.scaler,
        )

        """
        Actor Value Loss
        """
        for _ in range(self.num_imagination_iterations):
            (
                imagined_features,
                imagined_actions,
                imagined_discount,
                imagined_reward,
                old_imagined_value,
            ) = self.collect_imagination_data(state)
            imagined_actor_dist = self.actor(
                imagined_features[:-1].reshape(-1, imagined_features.shape[-1]).detach()
            )
            imagined_log_probs = imagined_actor_dist.log_prob(
                imagined_actions[:-1].reshape(-1, imagined_actions.shape[-1])
            ).detach()
            for _ in range(self.num_actor_value_updates):
                (
                    imagined_value,
                    imagined_returns,
                    weights,
                    imagined_features_values,
                ) = self.compute_values_for_training_actor_and_value(
                    imagined_features, imagined_reward, imagined_discount
                )

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

                vf_loss = self.value_loss(
                    imagined_features_values.detach(),
                    weights.detach(),
                    imagined_returns.detach(),
                    self.vf,
                    log_keys,
                    old_imagined_value,
                )

                if self.use_actor_value_optimizer:
                    self.scaler.scale(actor_loss + vf_loss).backward()
                    update_network(
                        [self.actor, self.vf],
                        self.actor_value_optimizer,
                        self.actor_gradient_clip,
                        self.scaler,
                    )
                else:
                    self.scaler.scale(actor_loss).backward()
                    update_network(
                        self.actor,
                        self.actor_optimizer,
                        self.actor_gradient_clip,
                        self.scaler,
                    )
                    self.scaler.scale(vf_loss).backward()
                    update_network(
                        self.vf,
                        self.vf_optimizer,
                        self.value_gradient_clip,
                        self.scaler,
                    )
        self.scaler.update()

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
            for key, value in log_keys.items():
                eval_statistics[key] = value

            if self.num_imagination_iterations > 0:
                eval_statistics["Imagined Returns"] = imagined_returns.mean().item()
                eval_statistics["Imagined Rewards"] = imagined_reward.mean().item()

        return eval_statistics

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
    @torch.cuda.amp.autocast()
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
            raps_obs_indices = np.arange(
                10,
                obs.shape[1],
                10,
            )
            raps_obs_indices = np.concatenate([[0], raps_obs_indices])
            obs = obs[:, raps_obs_indices]
            actions = actions[:, raps_obs_indices]
            prefix = "validation/"
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        (
            _,
            _,
            _,
            _,
            image_dist,
            reward_dist,
            pred_discount_dist,
            _,
        ) = self.world_model(obs, actions)
        ob = obs.reshape(-1, *self.image_shape)
        rewards = rewards.reshape(-1, rewards.shape[-1])
        terminals = terminals.reshape(-1, terminals.shape[-1])
        reward_error = torch.nn.functional.mse_loss(rewards, reward_dist.mean).item()
        discount_error = torch.nn.functional.mse_loss(
            pred_discount_dist.mean, terminals
        ).item()
        image_error = torch.nn.functional.mse_loss(
            self.world_model.preprocess(obs), image_dist.mean
        ).item()

        self.eval_statistics[prefix + "Full Obs Image Error"] = image_error
        self.eval_statistics[prefix + "Full Obs Reward Error"] = reward_error
        self.eval_statistics[prefix + "Full Obs Discount Error"] = discount_error

        self.eval_statistics[
            prefix + "Full Obs Primitive Model Image Error"
        ] = image_error
        self.eval_statistics[prefix + "Full Obs Primitive Reward Error"] = reward_error
        self.eval_statistics[
            prefix + "Full Obs Primitive Discount Error"
        ] = discount_error

        self.eval_statistics[prefix + "RAPS Obs Image Error"] = image_error
        self.eval_statistics[prefix + "RAPS Obs Reward Error"] = reward_error
        self.eval_statistics[prefix + "RAPS Obs Discount Error"] = discount_error

        self.eval_statistics[
            prefix + "RAPS Obs Primitive Model Image Error"
        ] = image_error
        self.eval_statistics[prefix + "RAPS Obs Primitive Reward Error"] = reward_error
        self.eval_statistics[
            prefix + "RAPS Obs Primitive Discount Error"
        ] = discount_error

    def save(self, path, suffix):
        actor = self.actor
        vf = self.vf
        world_model = self.world_model
        target_vf = self.target_vf
        actor_optimizer = self.actor_optimizer
        vf_optimizer = self.vf_optimizer
        world_model_optimizer = self.world_model_optimizer
        scaler = self.scaler
        actor_value_optimizer = self.actor_value_optimizer
        actor_entropy_loss_scale = self.actor_entropy_loss_scale

        delattr(self, "actor")
        delattr(self, "vf")
        delattr(self, "world_model")
        delattr(self, "target_vf")
        delattr(self, "actor_optimizer")
        delattr(self, "vf_optimizer")
        delattr(self, "world_model_optimizer")
        delattr(self, "scaler")
        delattr(self, "actor_value_optimizer")
        delattr(self, "actor_entropy_loss_scale")

        if hasattr(self, "buffer"):
            delattr(self, "buffer")

        pickle.dump(self, open(os.path.join(path, suffix), "wb"))

        base_suffix = suffix.replace(".pkl", "")
        torch.save(
            {
                "actor_state_dict": actor.state_dict(),
                "vf_state_dict": vf.state_dict(),
                "world_model_state_dict": world_model.state_dict(),
                "target_vf_state_dict": target_vf.state_dict(),
                "actor_optimizer_state_dict": actor_optimizer.state_dict(),
                "vf_optimizer_state_dict": vf_optimizer.state_dict(),
                "world_model_optimizer_state_dict": world_model_optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "actor_value_optimizer_state_dict": actor_value_optimizer.state_dict(),
            },
            os.path.join(path, base_suffix + "_networks_and_optimizers.ptc"),
        )

        self.actor = actor
        self.vf = vf
        self.world_model = world_model
        self.target_vf = target_vf
        self.actor_optimizer = actor_optimizer
        self.vf_optimizer = vf_optimizer
        self.world_model_optimizer = world_model_optimizer
        self.scaler = scaler
        self.actor_value_optimizer = actor_value_optimizer
        self.actor_entropy_loss_scale = actor_entropy_loss_scale

    def load(self, path, suffix):
        trainer = pickle.load(open(os.path.join(path, suffix), "rb"))
        trainer.actor = self.actor
        trainer.vf = self.vf
        trainer.world_model = self.world_model
        trainer.target_vf = self.target_vf
        trainer.actor_optimizer = self.actor_optimizer
        trainer.vf_optimizer = self.vf_optimizer
        trainer.world_model_optimizer = self.world_model_optimizer
        trainer.scaler = self.scaler
        trainer.actor_value_optimizer = self.actor_value_optimizer

        base_suffix = suffix.replace(".pkl", "")
        checkpoint = torch.load(
            os.path.join(path, base_suffix + "_networks_and_optimizers.ptc")
        )
        trainer.actor.load_state_dict(checkpoint["actor_state_dict"])
        trainer.vf.load_state_dict(checkpoint["vf_state_dict"])
        trainer.world_model.load_state_dict(checkpoint["world_model_state_dict"])
        trainer.target_vf.load_state_dict(checkpoint["target_vf_state_dict"])
        trainer.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state_dict"]
        )
        trainer.vf_optimizer.load_state_dict(checkpoint["vf_optimizer_state_dict"])
        trainer.world_model_optimizer.load_state_dict(
            checkpoint["world_model_optimizer_state_dict"]
        )
        trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        trainer.actor_value_optimizer.load_state_dict(
            checkpoint["actor_value_optimizer_state_dict"]
        )
        trainer.actor_entropy_loss_scale = (
            lambda x=self.actor_entropy_loss_schedule: schedule(
                x, self._n_train_steps_total
            )
        )

        return trainer


class DreamerV2LowLevelRAPSTrainer(DreamerV2Trainer):
    def __init__(
        self,
        *args,
        num_low_level_actions_per_primitive,
        batch_length,
        effective_batch_size_iterations,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_low_level_actions_per_primitive = num_low_level_actions_per_primitive
        self.batch_length = batch_length
        self.primitive_model_loss_function = nn.MSELoss()
        self.effective_batch_size_iterations = effective_batch_size_iterations

    def compute_world_model_loss_and_state(self, batch, log_keys):
        """
        :param batch: Dict
            rewards: (batch_size, path_length+1, 1)
            terminals: (batch_size, path_length+1, 1)
            observations: (batch_size, path_length+1, obs_dim)
            actions: (batch_size, path_length+1, act_dim)
        :log_keys: Counter

        :return state: Dict
            stoch: torch.Tensor (batch_size, path_length+1, stoch_size)
            mean: torch.Tensor (batch_size, path_length+1, stoch_size)
            std: torch.Tensor (batch_size, path_length+1, stoch_size)
            deter: torch.Tensor (batch_size, path_length+1, deter_size)
        """
        max_path_length = batch["observations"].shape[1]
        batch_size = batch["observations"].shape[0]
        raps_obs_indices = np.arange(
            self.num_low_level_actions_per_primitive,
            max_path_length,
            self.num_low_level_actions_per_primitive,
        )
        raps_obs_indices = np.concatenate([[0], raps_obs_indices])
        low_level_action_dim = batch["low_level_actions"].shape[-1]
        for itr in range(self.effective_batch_size_iterations):
            batch_indices = get_batch_length_indices(
                max_path_length, self.batch_length, batch_size
            )
            batch = self.buffer.random_batch(batch_size)
            rewards = ptu.from_numpy(batch["rewards"])
            terminals = ptu.from_numpy(batch["terminals"])
            obs = ptu.from_numpy(batch["observations"])
            high_level_actions = ptu.from_numpy(batch["high_level_actions"])
            low_level_actions = ptu.from_numpy(batch["low_level_actions"])
            log_keys["Reward in Batch"] += rewards.sum().item()
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
                    use_network_action=False,
                    batch_indices=batch_indices,
                    raps_obs_indices=raps_obs_indices,
                )
                obs = get_indexed_arr_from_batch_indices(obs, batch_indices).reshape(
                    -1, *self.image_shape
                )
                rewards = rewards.reshape(-1, rewards.shape[-1])
                terminals = terminals.reshape(-1, terminals.shape[-1])
                world_model_loss = self.world_model_loss(
                    image_dist,
                    reward_dist,
                    {
                        key: get_indexed_arr_from_batch_indices(value, batch_indices)
                        for key, value in prior.items()
                    },
                    {
                        key: get_indexed_arr_from_batch_indices(value, batch_indices)
                        for key, value in post.items()
                    },
                    prior_dist,
                    post_dist,
                    pred_discount_dist,
                    obs,
                    rewards,
                    terminals,
                    log_keys,
                )

                batch_indices_primitive_model = get_batch_length_indices(
                    action_preds.shape[1], action_preds.shape[1], batch_size
                )

                action_preds = get_indexed_arr_from_batch_indices(
                    action_preds, batch_indices_primitive_model
                ).reshape(-1, low_level_action_dim)
                low_level_actions = get_indexed_arr_from_batch_indices(
                    low_level_actions[:, 1:], batch_indices_primitive_model
                ).reshape(-1, low_level_action_dim)
                assert (
                    action_preds.shape == low_level_actions.shape
                ), f"Action Preds Shape: {action_preds.shape} != Low Level actions: {low_level_actions.shape}"
                primitive_loss = self.primitive_model_loss_function(
                    action_preds, low_level_actions
                )
                loss = (
                    world_model_loss + primitive_loss
                ) / self.effective_batch_size_iterations
                log_keys["Primitive Model Loss"] += primitive_loss.item()
            self.scaler.scale(loss).backward()
            if itr == 0:
                state = {
                    key: value[:, raps_obs_indices].detach()
                    for key, value in post.items()
                }
            else:
                for key, value in state.items():
                    state[key] = torch.cat(
                        [value, post[key][:, raps_obs_indices].detach()]
                    )
        for key in log_keys:
            log_keys[key] /= self.effective_batch_size_iterations
        return state

    def imagine_ahead(self, state, actor=None):
        """
        Imagine forward using actor and world model (no observations provided).

        :param state Dict
            stoch: torch.Tensor (batch_size, path_length+1, stoch_size)
            mean: torch.Tensor (batch_size, path_length+1, stoch_size)
            std: torch.Tensor (batch_size, path_length+1, stoch_size)
            deter: torch.Tensor (batch_size, path_length+1, deter_size)
        :param actor: torch.nn.Module

        :return imagined_features: torch.Tensor(imagination_horizon, batch_size*path_length, stoch_size + deter_size)
        :return imagined_actions: torch.Tensor(imagination_horizon, batch_size*path_length, action_size)
        """
        for k, v in state.items():
            assert (
                len(v.shape) == 3
            ), f"{k} should be of shape (batch_size, path_length+1, size). Got {v.shape}."
        torch.set_grad_enabled(self.use_dynamics_backprop)
        if actor is None:
            actor = self.actor
        new_state = {}
        for key, value in state.items():
            if self.use_pred_discount:
                value = value[:, :-1]
            new_state[key] = value.reshape(-1, value.shape[-1]).detach()
        imagined_features = []
        imagined_actions = []
        new_states = dict(mean=[], std=[], stoch=[], deter=[])
        for _ in range(self.imagination_horizon):
            features = self.world_model.get_features(new_state)
            for key in new_states.keys():
                new_states[key].append(new_state[key].unsqueeze(0))
            action_dist = actor(features.detach())
            high_level_action = action_dist.rsample()
            new_state, _ = self.world_model.forward_high_level_step(
                new_state,
                torch.empty(1),  # Dummy just to run function.
                torch.empty(1),  # Dummy just to run function.
                self.num_low_level_actions_per_primitive,
                high_level_action,
                use_obs=False,
                use_true_actions=False,
            )
            imagined_features.append(features.unsqueeze(0))
            imagined_actions.append(high_level_action.unsqueeze(0))
        imagined_features = torch.cat(imagined_features)
        imagined_actions = torch.cat(imagined_actions)
        for key in new_states.keys():
            new_states[key] = torch.cat(new_states[key])
        torch.set_grad_enabled(True)
        assert (
            imagined_features.shape[:2] == imagined_actions.shape[:2]
        ), "Imagined features and actions should have the same first two dimensions."
        assert (
            imagined_features.shape[0]
            == imagined_actions.shape[0]
            == self.imagination_horizon
        ), "Imagined dim 0 should be horizon."
        return imagined_features, imagined_actions, new_states

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def evaluate_helper(
        self,
        rewards,
        terminals,
        obs,
        target_obs,
        high_level_actions,
        low_level_actions,
        prefix,
        raps_obs_indices,
        use_network_action,
    ):
        (
            _,
            _,
            _,
            _,
            image_dist,
            reward_dist,
            pred_discount_dist,
            _,
            action_preds,
        ) = self.world_model(
            obs,
            (high_level_actions, low_level_actions),
            use_network_action=use_network_action,
            batch_indices=raps_obs_indices,
            raps_obs_indices=raps_obs_indices,
        )
        obs = target_obs.reshape(-1, *self.image_shape)
        rewards = rewards.reshape(-1, rewards.shape[-1])
        terminals = terminals.reshape(-1, terminals.shape[-1])
        primitive_loss = self.primitive_model_loss_function(
            action_preds.reshape(-1, action_preds.shape[-1]),
            low_level_actions[:, 1:].reshape(-1, action_preds.shape[-1]),
        )
        self.eval_statistics[prefix + "Image Error"] = torch.nn.functional.mse_loss(
            self.world_model.preprocess(obs), image_dist.mean
        ).item()
        self.eval_statistics[prefix + "Reward Error"] = torch.nn.functional.mse_loss(
            rewards, reward_dist.mean
        ).item()
        self.eval_statistics[prefix + "Discount Error"] = torch.nn.functional.mse_loss(
            pred_discount_dist.mean, terminals
        ).item()
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
        raps_obs_indices = np.arange(
            self.num_low_level_actions_per_primitive,
            obs.shape[1],
            self.num_low_level_actions_per_primitive,
        )
        raps_obs_indices = np.concatenate([[0], raps_obs_indices])
        self.evaluate_helper(
            rewards,
            terminals,
            obs,
            obs[:, raps_obs_indices],
            high_level_actions,
            low_level_actions,
            prefix + "Full Obs ",
            raps_obs_indices,
            use_network_action=False,
        )

        self.evaluate_helper(
            rewards,
            terminals,
            obs,
            obs[:, raps_obs_indices],
            high_level_actions,
            low_level_actions,
            prefix + "Full Obs Primitive Model ",
            raps_obs_indices,
            use_network_action=True,
        )

        self.evaluate_helper(
            rewards,
            terminals,
            obs[:, raps_obs_indices],
            obs[:, raps_obs_indices],
            high_level_actions,
            low_level_actions,
            prefix + "RAPS Obs ",
            raps_obs_indices,
            use_network_action=False,
        )

        self.evaluate_helper(
            rewards,
            terminals,
            obs[:, raps_obs_indices],
            obs[:, raps_obs_indices],
            high_level_actions,
            low_level_actions,
            prefix + "RAPS Obs Primitive Model ",
            raps_obs_indices,
            use_network_action=False,
        )
