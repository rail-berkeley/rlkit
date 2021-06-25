from collections import Counter, OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.core.loss import LossStatistics
from rlkit.torch.model_based.dreamer.dreamer_v2 import DreamerV2Trainer
from rlkit.torch.model_based.dreamer.utils import FreezeParameters, lambda_return

Plan2ExploreLosses = namedtuple(
    "Plan2ExploreLosses",
    "actor_loss vf_loss world_model_loss one_step_ensemble_loss exploration_actor_loss exploration_vf_loss",
)


class Plan2ExploreTrainer(DreamerV2Trainer):
    def __init__(
        self,
        env,
        actor,
        vf,
        target_vf,
        world_model,
        image_shape,
        exploration_actor,
        exploration_vf,
        exploration_target_vf,
        one_step_ensemble,
        exploration_intrinsic_reward_scale=1.0,
        exploration_extrinsic_reward_scale=0.0,
        evaluation_intrinsic_reward_scale=0.0,
        evaluation_extrinsic_reward_scale=1.0,
        log_disagreement=True,
        ensemble_training_states="post_to_next_post",
        **kwargs,
    ):
        super(Plan2ExploreTrainer, self).__init__(
            env,
            actor,
            vf,
            target_vf,
            world_model,
            image_shape,
            **kwargs,
        )
        self.exploration_actor = exploration_actor.to(ptu.device)
        self.exploration_vf = exploration_vf.to(ptu.device)
        self.exploration_target_vf = exploration_target_vf.to(ptu.device)
        self.one_step_ensemble = one_step_ensemble.to(ptu.device)

        self.one_step_ensemble_optimizer = self.optimizer_class(
            self.one_step_ensemble.parameters(),
            lr=self.world_model_lr,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )

        self.exploration_actor_optimizer = self.optimizer_class(
            self.exploration_actor.parameters(),
            lr=self.actor_lr,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )
        self.exploration_vf_optimizer = self.optimizer_class(
            self.exploration_vf.parameters(),
            lr=self.vf_lr,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )
        self.exploration_intrinsic_reward_scale = exploration_intrinsic_reward_scale
        self.exploration_extrinsic_reward_scale = exploration_extrinsic_reward_scale
        self.evaluation_intrinsic_reward_scale = evaluation_intrinsic_reward_scale
        self.evaluation_extrinsic_reward_scale = evaluation_extrinsic_reward_scale
        self.log_disagreement = log_disagreement
        self.ensemble_training_states = ensemble_training_states

    def update_target_networks(self):
        super().update_target_networks()
        ptu.soft_update_from_to(
            self.exploration_vf, self.exploration_target_vf, self.soft_target_tau
        )

    def compute_exploration_reward(
        self, exploration_imag_states, exploration_imag_actions
    ):
        pred_embeddings = []
        d = {
            "deter": exploration_imag_states["deter"],
            "stoch": exploration_imag_states["stoch"],
            "feat": self.world_model.get_features(exploration_imag_states),
        }
        input_state = d[self.one_step_ensemble.inputs]
        input_state = input_state.reshape(-1, input_state.shape[-1])
        exploration_imag_actions = exploration_imag_actions.reshape(
            -1, exploration_imag_actions.shape[-1]
        )
        inputs = torch.cat((input_state, exploration_imag_actions), 1)
        for mdl in range(self.one_step_ensemble.num_models):
            pred_embeddings.append(
                self.one_step_ensemble.forward_ith_model(inputs, mdl).mean.unsqueeze(0)
            )
        pred_embeddings = torch.cat(pred_embeddings)

        # computes std across ensembles, squares it to compute variance and then computes the mean across the vector dim
        reward = (pred_embeddings.std(dim=0)).mean(dim=-1)
        if self.log_disagreement:
            reward = torch.log(reward)
        return reward

    def train_networks(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[Plan2ExploreLosses, LossStatistics]:
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        """
        World Model Loss
        """
        with torch.cuda.amp.autocast():

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

            obs = obs.transpose(1, 0).reshape(-1, np.prod(self.image_shape))
            rewards = rewards.transpose(1, 0).reshape(-1, rewards.shape[-1])
            terminals = terminals.transpose(1, 0).reshape(-1, terminals.shape[-1])
            actions = actions[:, :-1, :].transpose(1, 0).reshape(-1, actions.shape[-1])
            post_vals = {
                "embed": embed,
                "stoch": post["stoch"],
                "deter": post["deter"],
                "feat": self.world_model.get_features(post),
            }
            prior_vals = {
                "embed": embed,
                "stoch": prior["stoch"],
                "deter": prior["deter"],
                "feat": self.world_model.get_features(prior),
            }
            ensemble_training_input_target_state = {
                "post_to_next_post": [post_vals, post_vals],
                "prior_to_next_prior": [prior_vals, prior_vals],
                "post_to_next_prior": [post_vals, prior_vals],
                "prior_to_next_post": [prior_vals, post_vals],
            }
            input_vals, target_vals = ensemble_training_input_target_state[
                self.ensemble_training_states
            ]
            one_step_ensemble_inputs = input_vals[self.one_step_ensemble.inputs]
            one_step_ensemble_targets = target_vals[self.one_step_ensemble.targets]

            one_step_ensemble_inputs = (
                one_step_ensemble_inputs[:, :-1, :]
                .transpose(1, 0)
                .reshape(-1, one_step_ensemble_inputs.shape[-1])
            )
            one_step_ensemble_inputs = torch.cat(
                (one_step_ensemble_inputs, actions), -1
            ).detach()
            one_step_ensemble_targets = (
                one_step_ensemble_targets[:, 1:, :]
                .transpose(1, 0)
                .reshape(-1, one_step_ensemble_targets.shape[-1])
            ).detach()

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
            self.world_model_gradient_clip,
        )

        """
        One Step Ensemble Loss
        """
        with torch.cuda.amp.autocast():
            ensemble_loss = 0
            for mdl in range(self.one_step_ensemble.num_models):
                member_pred = self.one_step_ensemble.forward_ith_model(
                    one_step_ensemble_inputs, mdl
                )  # predict embedding of next state
                member_loss = (
                    -1 * member_pred.log_prob(one_step_ensemble_targets).mean()
                )
                ensemble_loss += member_loss

        self.update_network(
            self.one_step_ensemble,
            self.one_step_ensemble_optimizer,
            ensemble_loss,
            self.world_model_gradient_clip,
        )

        """
        Actor Value Loss
        """
        world_model_params = list(self.world_model.parameters())
        vf_params = list(self.vf.parameters())
        target_vf_params = list(self.target_vf.parameters())
        one_step_ensemble_params = list(self.one_step_ensemble.parameters())
        exploration_vf_params = list(self.exploration_vf.parameters())
        exploration_target_vf_params = list(self.exploration_target_vf.parameters())
        pred_discount_params = list(self.world_model.pred_discount.parameters())
        log_keys = Counter()
        for _ in range(self.num_imagination_iterations):
            with torch.cuda.amp.autocast():
                with FreezeParameters(
                    world_model_params + one_step_ensemble_params + pred_discount_params
                ):
                    (
                        imag_feat,
                        imag_actions,
                        imag_states,
                    ) = self.imagine_ahead(post, actor=self.actor)
                    intrinsic_reward = self.compute_exploration_reward(
                        imag_states, imag_actions
                    )
                    extrinsic_reward = self.world_model.reward(imag_feat)
                    intrinsic_reward = torch.cat(
                        [
                            intrinsic_reward[i : i + imag_feat.shape[1]]
                            .unsqueeze(0)
                            .unsqueeze(2)
                            for i in range(
                                0,
                                intrinsic_reward.shape[0],
                                imag_feat.shape[1],
                            )
                        ],
                        0,
                    )
                    imag_reward = (
                        intrinsic_reward * self.evaluation_intrinsic_reward_scale
                        + extrinsic_reward * self.evaluation_extrinsic_reward_scale
                    )
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
                imag_features_actions = (
                    imag_feat[:-1].reshape(-1, imag_feat.shape[-1]).detach()
                )
                imag_actions_actions = (
                    imag_actions[:-1].reshape(-1, imag_actions.shape[-1]).detach()
                )
                imag_actor_dist = self.actor(imag_features_actions)
                imag_log_probs = imag_actor_dist.log_prob(imag_actions_actions).detach()
            for _ in range(self.num_actor_value_updates):
                with torch.cuda.amp.autocast():
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

                    actor_loss = self.actor_loss(
                        imag_returns,
                        imag_value,
                        imag_feat,
                        imag_actions,
                        weights,
                        imag_log_probs,
                        self.actor,
                        log_keys,
                    )

                    with torch.no_grad():
                        imag_features_v = imag_feat.detach()
                        target = imag_returns.detach()
                        weights = weights.detach()

                    vf_loss = self.value_loss(
                        imag_features_v,
                        weights,
                        target,
                        self.vf,
                        log_keys,
                        old_imag_value,
                    )

                if self.use_actor_value_optimizer:
                    self.update_network(
                        [self.actor, self.vf],
                        self.actor_value_optimizer,
                        actor_loss + vf_loss,
                        self.actor_gradient_clip,
                    )
                else:
                    self.update_network(
                        self.actor,
                        self.actor_optimizer,
                        actor_loss,
                        self.actor_gradient_clip,
                    )

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
        Exploration Actor Loss
        """
        for _ in range(self.num_imagination_iterations):
            with torch.cuda.amp.autocast():
                with FreezeParameters(
                    world_model_params + one_step_ensemble_params + pred_discount_params
                ):
                    (
                        exploration_imag_feat,
                        exploration_imag_actions,
                        exploration_imag_states,
                    ) = self.imagine_ahead(
                        post,
                        actor=self.exploration_actor,
                    )
                    exploration_intrinsic_reward = self.compute_exploration_reward(
                        exploration_imag_states, exploration_imag_actions
                    )
                    exploration_extrinsic_reward = self.world_model.reward(
                        exploration_imag_feat
                    )
                    exploration_intrinsic_reward = torch.cat(
                        [
                            exploration_intrinsic_reward[
                                i : i + exploration_imag_feat.shape[1]
                            ]
                            .unsqueeze(0)
                            .unsqueeze(2)
                            for i in range(
                                0,
                                exploration_intrinsic_reward.shape[0],
                                exploration_imag_feat.shape[1],
                            )
                        ],
                        0,
                    )
                    exploration_imag_reward = (
                        exploration_intrinsic_reward
                        * self.exploration_intrinsic_reward_scale
                        + exploration_extrinsic_reward
                        * self.exploration_extrinsic_reward_scale
                    )

                    if self.use_pred_discount:
                        exploration_discount = self.world_model.get_dist(
                            self.world_model.pred_discount(exploration_imag_feat),
                            std=None,
                            normal=False,
                        ).mean
                    else:
                        exploration_discount = self.discount * torch.ones_like(
                            exploration_imag_reward
                        )
                with FreezeParameters(vf_params):
                    exploration_old_imag_value = self.exploration_vf(
                        exploration_imag_feat
                    ).detach()

                exploration_imag_features_actions = (
                    exploration_imag_feat[:-1]
                    .reshape(-1, exploration_imag_feat.shape[-1])
                    .detach()
                )
                exploration_imag_actions_actions = (
                    exploration_imag_actions[:-1]
                    .reshape(-1, exploration_imag_actions.shape[-1])
                    .detach()
                )
                exploration_imag_actor_dist = self.exploration_actor(
                    exploration_imag_features_actions
                )
                exploration_imag_log_probs = exploration_imag_actor_dist.log_prob(
                    exploration_imag_actions_actions
                ).detach()
            for _ in range(self.num_actor_value_updates):
                with torch.cuda.amp.autocast():
                    with FreezeParameters(
                        exploration_vf_params + exploration_target_vf_params
                    ):
                        exploration_imag_target_value = self.exploration_target_vf(
                            exploration_imag_feat
                        )
                        exploration_imag_value = self.exploration_vf(
                            exploration_imag_feat
                        )
                    exploration_imag_returns = lambda_return(
                        exploration_imag_reward[:-1],
                        exploration_imag_target_value[:-1],
                        exploration_discount[:-1],
                        bootstrap=exploration_imag_target_value[-1],
                        lambda_=self.lam,
                    )
                    exploration_weights = torch.cumprod(
                        torch.cat(
                            [
                                torch.ones_like(exploration_discount[:1]),
                                exploration_discount[:-1],
                            ],
                            0,
                        ),
                        0,
                    ).detach()[:-1]
                    exploration_actor_loss = self.actor_loss(
                        exploration_imag_returns,
                        exploration_imag_value,
                        exploration_imag_feat,
                        exploration_imag_actions,
                        exploration_weights,
                        exploration_imag_log_probs,
                        self.exploration_actor,
                        log_keys,
                        prefix="exploration_",
                    )

                    with torch.no_grad():
                        exploration_imag_features_v = exploration_imag_feat.detach()
                        exploration_value_target = exploration_imag_returns.detach()
                        exploration_weights = exploration_weights.detach()

                    exploration_vf_loss = self.value_loss(
                        exploration_imag_features_v,
                        exploration_weights,
                        exploration_value_target,
                        self.exploration_vf,
                        log_keys,
                        exploration_old_imag_value,
                        prefix="",
                    )

                if self.use_actor_value_optimizer:
                    self.update_network(
                        [self.exploration_actor, self.exploration_vf],
                        self.exploration_actor_value_optimizer,
                        exploration_actor_loss + exploration_vf_loss,
                        self.actor_gradient_clip,
                    )
                else:
                    self.update_network(
                        self.exploration_actor,
                        self.exploration_actor_optimizer,
                        exploration_actor_loss,
                        self.actor_gradient_clip,
                    )

                    self.update_network(
                        self.exploration_vf,
                        self.exploration_vf_optimizer,
                        exploration_vf_loss,
                        self.value_gradient_clip,
                    )

        if self.num_imagination_iterations > 0:
            for key in log_keys:
                log_keys[key] /= (
                    self.num_actor_value_updates * self.num_imagination_iterations
                )

        self.scaler.update()
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
            eval_statistics["Posterior State Std"] = post["std"].mean().item()
            eval_statistics["Prior State Std"] = prior["std"].mean().item()
            eval_statistics["One Step Ensemble Loss"] = ensemble_loss.item()

            eval_statistics["Actor Loss"] = log_keys["actor_loss"]
            eval_statistics["Dynamics Backprop Loss"] = log_keys[
                "dynamics_backprop_loss"
            ]
            eval_statistics["Policy Gradient Loss"] = log_keys["policy_gradient_loss"]
            eval_statistics["Actor Entropy Loss"] = log_keys["actor_entropy_loss"]
            eval_statistics["Actor Entropy Loss Scale"] = log_keys[
                "actor_entropy_loss_scale"
            ]
            eval_statistics["Actor Log Probs"] = log_keys["log_probs"]
            eval_statistics["Value Loss"] = log_keys["vf_loss"]

            eval_statistics["Exploration Actor Loss"] = log_keys[
                "exploration_actor_loss"
            ]
            eval_statistics["Exploration Dynamics Backprop Loss"] = log_keys[
                "exploration_dynamics_backprop_loss"
            ]
            eval_statistics["Exploration Policy Gradient Loss"] = log_keys[
                "exploration_policy_gradient_loss"
            ]
            eval_statistics["Exploration Actor Entropy Loss"] = log_keys[
                "exploration_actor_entropy_loss"
            ]
            eval_statistics["Exploration Actor Entropy Loss Scale"] = log_keys[
                "exploration_actor_entropy_loss_scale"
            ]
            eval_statistics["Exploration Actor Log Probs"] = log_keys[
                "exploration_log_probs"
            ]
            eval_statistics["Exploration Value Loss"] = log_keys["exploration_vf_loss"]

            eval_statistics["Imagined Returns"] = imag_returns.mean().item()
            eval_statistics["Imagined Rewards"] = imag_reward.mean().item()
            eval_statistics["Imagined Values"] = log_keys["imag_values_mean"]
            eval_statistics["Predicted Rewards"] = reward_dist.mean.mean().item()
            eval_statistics[
                "Imagined Intrinsic Rewards"
            ] = intrinsic_reward.mean().item()
            eval_statistics[
                "Imagined Extrinsic Rewards"
            ] = extrinsic_reward.mean().item()

            eval_statistics["Exploration Imagined Values"] = log_keys[
                "exploration_imag_values_mean"
            ]

            eval_statistics[
                "Exploration Imagined Returns"
            ] = exploration_imag_returns.mean().item()
            eval_statistics[
                "Exploration Imagined Rewards"
            ] = exploration_imag_reward.mean().item()
            eval_statistics[
                "Exploration Imagined Intrinsic Rewards"
            ] = exploration_intrinsic_reward.mean().item()
            eval_statistics[
                "Exploration Imagined Extrinsic Rewards"
            ] = exploration_extrinsic_reward.mean().item()

        loss = Plan2ExploreLosses(
            actor_loss=actor_loss,
            world_model_loss=world_model_loss,
            vf_loss=vf_loss,
            one_step_ensemble_loss=ensemble_loss,
            exploration_actor_loss=exploration_actor_loss,
            exploration_vf_loss=exploration_vf_loss,
        )

        return loss, eval_statistics

    @property
    def networks(self):
        return [
            self.actor,
            self.vf,
            self.target_vf,
            self.world_model,
            self.one_step_ensemble,
            self.exploration_actor,
            self.exploration_vf,
            self.exploration_target_vf,
        ]

    @property
    def optimizers(self):
        return [
            self.actor_optimizer,
            self.vf_optimizer,
            self.world_model_optimizer,
            self.one_step_ensemble_optimizer,
            self.exploration_actor_optimizer,
            self.exploration_vf_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            actor=self.actor,
            world_model=self.world_model,
            vf=self.vf,
            target_vf=self.target_vf,
            one_step_ensemble=self.one_step_ensemble,
            exploration_actor=self.exploration_actor,
            exploration_vf=self.exploration_vf,
            exploration_target_vf=self.exploration_target_vf,
        )
