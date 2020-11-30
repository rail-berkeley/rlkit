from collections import OrderedDict, namedtuple
from typing import Tuple

import gtimer as gt
import numpy as np
import torch
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.core.loss import LossStatistics
from rlkit.torch.model_based.dreamer.dreamer_v2 import DreamerV2Trainer
from rlkit.torch.model_based.dreamer.utils import FreezeParameters, lambda_return

try:
    import apex
    from apex import amp

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

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
        world_model,
        imagination_horizon,
        image_shape,
        exploration_actor,
        exploration_vf,
        one_step_ensemble,
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
        free_nats=3.0,
        kl_loss_scale=1.0,
        pred_discount_loss_scale=10.0,
        adam_eps=1e-7,
        weight_decay=0.0,
        use_pred_discount=True,
        debug=False,
        exploration_reward_scale=10000,
    ):
        super(Plan2ExploreTrainer, self).__init__(
            env,
            actor,
            vf,
            world_model,
            imagination_horizon,
            image_shape,
            target_vf=target_vf,
            discount=discount,
            reward_scale=reward_scale,
            actor_lr=actor_lr,
            vf_lr=vf_lr,
            world_model_lr=world_model_lr,
            optimizer_class=optimizer_class,
            use_amp=use_amp,
            opt_level=opt_level,
            gradient_clip=gradient_clip,
            lam=lam,
            free_nats=free_nats,
            kl_loss_scale=kl_loss_scale,
            pred_discount_loss_scale=pred_discount_loss_scale,
            adam_eps=adam_eps,
            weight_decay=weight_decay,
            use_pred_discount=use_pred_discount,
            debug=debug,
            image_loss_scale=1.0,
            reward_loss_scale=1.0,
            transition_loss_scale=0.0,
            entropy_loss_scale=0.0,
            forward_kl=True,
            reinforce_loss_scale=0.0,
            dynamics_backprop_loss_scale=1.0,
            actor_entropy_loss_schedule="0.0",
            soft_target_tau=1,
            target_update_period=1,
            initialize_amp=False,
        )
        self.exploration_actor = exploration_actor.to(ptu.device)
        self.exploration_vf = exploration_vf.to(ptu.device)
        self.one_step_ensemble = one_step_ensemble.to(ptu.device)

        self.one_step_ensemble_optimizer = self.optimizer_class(
            self.one_step_ensemble.parameters(),
            lr=world_model_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )

        self.exploration_actor_optimizer = self.optimizer_class(
            self.exploration_actor.parameters(),
            lr=actor_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        self.exploration_vf_optimizer = self.optimizer_class(
            self.exploration_vf.parameters(),
            lr=vf_lr,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
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
                    self.one_step_ensemble,
                    self.exploration_actor,
                    self.exploration_vf,
                ],
                [
                    self.world_model_optimizer,
                    self.actor_optimizer,
                    self.vf_optimizer,
                    self.one_step_ensemble_optimizer,
                    self.exploration_actor_optimizer,
                    self.exploration_vf_optimizer,
                ],
                opt_level=opt_level,
                num_losses=6,
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
                self.one_step_ensemble,
                self.exploration_actor,
                self.exploration_vf,
            ) = models
            (
                self.world_model_optimizer,
                self.actor_optimizer,
                self.vf_optimizer,
                self.one_step_ensemble_optimizer,
                self.exploration_actor_optimizer,
                self.exploration_vf_optimizer,
            ) = optimizers

        self.exploration_reward_scale = exploration_reward_scale

    def compute_exploration_reward(
        self, exploration_imag_deter_states, exploration_imag_actions
    ):
        pred_embeddings = []
        input_state = exploration_imag_deter_states
        exploration_imag_actions = torch.cat(
            [
                exploration_imag_actions[i, :, :]
                for i in range(exploration_imag_actions.shape[0])
            ]
        )
        for mdl in range(self.one_step_ensemble.num_models):
            inputs = torch.cat((input_state, exploration_imag_actions), 1)
            pred_embeddings.append(
                self.one_step_ensemble.forward_ith_model(inputs, mdl).mean.unsqueeze(0)
            )
        pred_embeddings = torch.cat(pred_embeddings)

        assert pred_embeddings.shape[0] == self.one_step_ensemble.num_models
        assert pred_embeddings.shape[1] == input_state.shape[0]
        assert len(pred_embeddings.shape) == 3

        reward = (pred_embeddings.std(dim=0) * pred_embeddings.std(dim=0)).mean(
            dim=1
        ) * self.exploration_reward_scale
        return reward

    def imagine_ahead(self, state, actor):
        new_state = {}
        for k, v in state.items():
            with torch.no_grad():
                if self.use_pred_discount:  # Last step could be terminal.
                    v = v[:, :-1]
                new_state[k] = torch.cat([v[:, i, :] for i in range(v.shape[1])])
        feats = []
        actions = []
        states = []
        for i in range(self.imagination_horizon):
            feat = self.world_model.get_feat(new_state).detach()
            action = actor(feat).rsample()
            new_state = self.world_model.img_step(new_state, action)
            feats.append(self.world_model.get_feat(new_state).unsqueeze(0))
            actions.append(action.unsqueeze(0))
            states.append(new_state["deter"])

        feats = torch.cat(feats)
        actions = torch.cat(actions)
        states = torch.cat(states)
        return feats, actions, states

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
        **kwargs,
    ) -> Tuple[Plan2ExploreLosses, LossStatistics]:
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
        actions = torch.cat([actions[:, i, :] for i in range(actions.shape[1])])
        embed = torch.cat([embed[:, i, :] for i in range(embed.shape[1])])
        deter = torch.cat(
            [
                prior["deter"][:, i, :] for i in range(prior["deter"].shape[1])
            ]  # should take prior deter because we need the value before the embedding is incorporated
        )
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
        target_vf_params = list(self.target_vf.parameters())
        one_step_ensemble_params = list(self.one_step_ensemble.parameters())
        exploration_vf_params = list(self.exploration_vf.parameters())
        pred_discount_params = list(self.world_model.pred_discount.parameters())

        with FreezeParameters(world_model_params):
            imag_feat, imag_actions, _ = self.imagine_ahead(post, actor=self.actor)
        with FreezeParameters(world_model_params + vf_params + target_vf_params):
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
        ) = self.actor_loss(
            imag_returns, self.vf(imag_feat), imag_feat, imag_actions, weights
        )

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
        One Step Ensemble Loss
        """
        batch_size = rewards.shape[0]
        bagging_size = int(1 * batch_size)
        indices = np.random.randint(
            low=0,
            high=batch_size,
            size=[self.one_step_ensemble.num_models, bagging_size],
        )
        ensemble_loss = 0

        for mdl in range(self.one_step_ensemble.num_models):
            mdl_actions = actions[indices[mdl, :]].detach()
            target_prediction = embed[indices[mdl, :]].detach()
            input_state = deter[indices[mdl, :]].detach()
            inputs = torch.cat((input_state, mdl_actions), 1)
            member_pred = self.one_step_ensemble.forward_ith_model(
                inputs, mdl
            )  # predict embedding of next state
            member_loss = -1 * member_pred.log_prob(target_prediction).mean()
            ensemble_loss += member_loss

        self.update_network(
            self.one_step_ensemble, self.one_step_ensemble_optimizer, ensemble_loss, 3
        )

        """
        Exploration Actor Loss
        """
        with FreezeParameters(world_model_params):
            (
                exploration_imag_feat,
                exploration_imag_actions,
                exploration_imag_deter_states,
            ) = self.imagine_ahead(
                post,
                actor=self.exploration_actor,
            )
        with FreezeParameters(
            world_model_params + exploration_vf_params + one_step_ensemble_params
        ):
            exploration_reward = self.compute_exploration_reward(
                exploration_imag_deter_states, exploration_imag_actions
            )  # Compute Intrinsic Reward

            exploration_reward = torch.cat(
                [
                    exploration_reward[i : i + exploration_imag_feat.shape[1]]
                    .unsqueeze(0)
                    .unsqueeze(2)
                    for i in range(
                        0, exploration_reward.shape[0], exploration_imag_feat.shape[1]
                    )
                ],
                0,
            )

            if self.use_pred_discount:
                with FreezeParameters(pred_discount_params):
                    exploration_discount = self.world_model.get_dist(
                        self.world_model.pred_discount(exploration_imag_feat),
                        std=None,
                        normal=False,
                    ).mean
            else:
                exploration_discount = self.discount * torch.ones_like(
                    exploration_reward
                )
            exploration_value = self.exploration_vf(exploration_imag_feat)
        assert (
            exploration_reward.shape == exploration_value.shape
            and exploration_reward.shape == exploration_discount.shape
        )
        exploration_returns = lambda_return(
            exploration_reward[:-1],
            exploration_value[:-1],
            exploration_discount[:-1],
            bootstrap=exploration_value[-1],
            lambda_=self.lam,
        )
        exploration_weights = torch.cumprod(
            torch.cat(
                [torch.ones_like(exploration_discount[:1]), exploration_discount[:-2]],
                0,
            ),
            0,
        ).detach()
        exploration_actor_loss = -(exploration_weights * exploration_returns).mean()

        self.update_network(
            self.exploration_actor,
            self.exploration_actor_optimizer,
            exploration_actor_loss,
            4,
        )
        """
        Exploration Value Loss
        """
        with torch.no_grad():
            exploration_imag_feat_v = exploration_imag_feat.detach()
            exploration_value_target = exploration_returns.detach()
            exploration_weights = exploration_weights.detach()

        exploration_vf_loss, exploration_value_dist = self.value_loss(
            exploration_imag_feat_v,
            exploration_weights,
            exploration_value_target,
            vf=self.exploration_vf,
        )

        self.update_network(
            self.exploration_vf, self.exploration_vf_optimizer, exploration_vf_loss, 5
        )

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics["Value Loss"] = vf_loss.item()
            eval_statistics["Actor Loss"] = actor_loss.item()
            eval_statistics["World Model Loss"] = world_model_loss.item()
            eval_statistics["Image Loss"] = image_pred_loss.item()
            eval_statistics["Reward Loss"] = reward_pred_loss.item()
            eval_statistics["Divergence Loss"] = div.item()
            if self.use_pred_discount:
                eval_statistics["Pred Discount Loss"] = pred_discount_loss.item()

            eval_statistics["Dynamics Backprop Loss"] = dynamics_backprop_loss.item()
            eval_statistics["Reinforce Loss"] = reinforce_loss.item()
            eval_statistics["Actor Entropy Loss"] = actor_entropy_loss.item()
            eval_statistics["Actor Entropy Loss Scale"] = actor_entropy_loss_scale

            eval_statistics["Imagined Returns"] = imag_returns.mean().item()
            eval_statistics["Imagined Rewards"] = imag_reward.mean().item()
            eval_statistics["Imagined Values"] = value_dist.mean.mean().item()
            eval_statistics["Predicted Rewards"] = reward_dist.mean.mean().item()

            eval_statistics["One Step Ensemble Loss"] = ensemble_loss.item()
            eval_statistics["Exploration Value Loss"] = exploration_vf_loss.item()
            eval_statistics["Exploration Actor Loss"] = exploration_actor_loss.item()
            eval_statistics[
                "Exploration Imagined Values"
            ] = exploration_value_dist.mean.mean().item()

            eval_statistics[
                "Exploration Imagined Returns"
            ] = exploration_returns.mean().item()
            eval_statistics["Imagined Rewards"] = exploration_reward.mean().item()
            eval_statistics[
                "Imagined Values"
            ] = exploration_value_dist.mean.mean().item()

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
            self.world_model,
            self.one_step_ensemble,
            self.exploration_actor,
            self.exploration_vf,
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
            one_step_ensemble=self.one_step_ensemble,
            exploration_actor=self.exploration_actor,
            exploration_vf=self.exploration_vf,
        )
