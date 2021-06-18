from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.core.loss import LossStatistics
from rlkit.torch.model_based.dreamer.actor_models import ConditionalActorModel
from rlkit.torch.model_based.dreamer.dreamer_v2 import DreamerV2Trainer
from rlkit.torch.model_based.dreamer.utils import FreezeParameters, lambda_return
from rlkit.torch.model_based.plan2explore.advanced_mcts_wm_expl import (
    Advanced_UCT_search,
)

DreamerLosses = namedtuple(
    "DreamerLosses",
    "actor_loss vf_loss world_model_loss",
)


class DreamerV2MCTSTrainer(DreamerV2Trainer):
    def __init__(
        self,
        *args,
        mcts_kwargs=None,
        **kwargs,
    ):
        super(DreamerV2MCTSTrainer, self).__init__(*args, **kwargs)

        self.mcts_kwargs = mcts_kwargs

    def actor_loss(
        self,
        imag_returns,
        value,
        imag_feat,
        imag_actions,
        weights,
        old_imag_log_probs,
        actor=None,
    ):
        if actor is None:
            actor = self.actor
        assert len(imag_returns.shape) == 3, imag_returns.shape
        assert len(value.shape) == 3, value.shape
        assert len(imag_feat.shape) == 3, imag_feat.shape
        assert len(imag_actions.shape) == 3, imag_actions.shape
        assert len(weights.shape) == 3 and weights.shape[-1] == 1, weights.shape
        if self.actor.use_tanh_normal:
            assert imag_actions.max() <= 1.0 and imag_actions.min() >= -1.0
        if self.use_baseline:
            advantages = imag_returns - value[:-1]
        else:
            advantages = imag_returns
        if self.use_advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        advantages = (
            torch.cat([advantages[i, :, :] for i in range(advantages.shape[0])])
            .detach()
            .squeeze(-1)
        )

        imag_feat_a = torch.cat(
            [imag_feat[i, :, :] for i in range(imag_feat.shape[0] - 1)]
        ).detach()
        imag_actions = torch.cat(
            [imag_actions[i, :, :] for i in range(imag_actions.shape[0] - 1)]
        ).detach()
        old_imag_log_probs = torch.cat(
            [old_imag_log_probs[i, :] for i in range(old_imag_log_probs.shape[0] - 1)]
        ).detach()
        assert imag_actions.shape[0] == imag_feat_a.shape[0]

        imag_discrete_actions = imag_actions[:, : self.world_model.env.num_primitives]
        action_input = (imag_discrete_actions, imag_feat_a.detach())
        imag_actor_dist = actor(action_input)
        imag_continuous_actions = imag_actions[:, self.world_model.env.num_primitives :]
        imag_log_probs = imag_actor_dist.log_prob(imag_continuous_actions)
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

    def imagine_ahead(self, state, discrete_actions):
        new_state = {}
        for k, v in state.items():
            with torch.no_grad():
                if self.use_pred_discount:  # Last step could be terminal.
                    v = v[:, :-1]
                new_state[k] = v.transpose(1, 0).reshape(-1, v.shape[-1])
        feats = []
        actions = []
        log_probs = []
        for i in range(self.imagination_horizon):
            feat = self.world_model.get_features(new_state)
            discrete_action = ptu.from_numpy(
                discrete_actions[i : i + 1, : self.world_model.env.num_primitives]
            ).repeat(feat.shape[0], 1)
            action_input = (discrete_action, feat.detach())
            action_dist = self.actor(action_input)
            if type(self.actor) == ConditionalActorModel:
                continuous_action, log_prob = action_dist.rsample_and_log_prob()
            else:
                continuous_action = action_dist.rsample()
                log_prob = action_dist.log_prob(continuous_action)
            action = torch.cat((discrete_action, continuous_action), 1)
            new_state = self.world_model.action_step(new_state, action)

            feats.append(feat.unsqueeze(0))
            actions.append(action.unsqueeze(0))
            log_probs.append(log_prob.unsqueeze(0))
        feats = torch.cat(feats)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        return feats, actions, log_probs

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
            o = self.world_model.env.reset()
            o = np.concatenate([o.reshape(1, -1) for i in range(1)])
            latent = self.world_model.initial(1)
            action = ptu.zeros((1, self.world_model.env.action_space.low.size))
            o = ptu.from_numpy(np.array(o))
            embed = self.world_model.encode(o)
            start_state, _ = self.world_model.obs_step(latent, action, embed)
            discrete_actions = Advanced_UCT_search(
                self.world_model,
                None,
                self.actor,
                start_state,
                self.world_model.env.num_primitives,
                self.vf,
                return_open_loop_plan=True,
                **self.mcts_kwargs,
            )
            with FreezeParameters(world_model_params + pred_discount_params):
                (
                    imag_feat,
                    imag_actions,
                    imag_log_probs,
                ) = self.imagine_ahead(post, discrete_actions)
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
                    imag_log_probs,
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
