from collections import OrderedDict
import copy
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
from torch.distributions import kl_divergence

import rlkit.torch.pytorch_util as ptu

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from itertools import chain


class PEARLSoftActorCriticTrainer(TorchTrainer):
    def __init__(
            self,
            latent_dim,
            agent,
            qf1,
            qf2,
            vf,
            context_encoder,
            reward_predictor,
            context_decoder,

            reward_scale=1.,
            discount=0.99,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,
            train_context_decoder=False,
            backprop_q_loss_into_encoder=True,

            train_reward_pred_in_unsupervised_phase=False,
            use_encoder_snapshot_for_reward_pred_in_unsupervised_phase=False,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,
    ):
        super().__init__()

        self.train_agent = True
        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        assert target_update_period == 1
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.train_reward_pred_in_unsupervised_phase = train_reward_pred_in_unsupervised_phase
        self.use_encoder_snapshot_for_reward_pred_in_unsupervised_phase = (
            use_encoder_snapshot_for_reward_pred_in_unsupervised_phase
        )

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.reward_pred_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context
        self.train_encoder_decoder = True
        self.train_context_decoder = train_context_decoder
        self.backprop_q_loss_into_encoder = backprop_q_loss_into_encoder

        self.agent = agent
        self.policy = agent.policy
        self.qf1, self.qf2, self.vf = qf1, qf2, vf
        self.target_vf = copy.deepcopy(self.vf)
        self.context_encoder = context_encoder
        self.context_decoder = context_decoder
        self.reward_predictor = reward_predictor

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        if train_context_decoder:
            self.context_optimizer = optimizer_class(
                chain(
                    self.context_encoder.parameters(),
                    self.context_decoder.parameters(),
                ),
                lr=context_lr,
            )
        else:
            self.context_optimizer = optimizer_class(
                self.context_encoder.parameters(),
                lr=context_lr,
            )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.reward_predictor_optimizer = optimizer_class(
            self.reward_predictor.parameters(),
            lr=context_lr,
        )

        self.eval_statistics = None
        self._need_to_update_eval_statistics = True

    ###### Torch stuff #####
    @property
    def networks(self):
        return [
            self.policy,
            self.qf1, self.qf2, self.vf, self.target_vf,
            self.context_encoder,
            self.context_decoder,
            self.reward_predictor,
        ]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)


    ##### Training #####
    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    # def train_from_torch(self, indices, context, context_dict):
    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        context = batch['context']

        # data is (task, batch, feat)
        # obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        action_distrib, p_z, task_z_with_grad = self.agent(
            obs, context, return_latent_posterior_and_task_z=True,
        )
        task_z_detached = task_z_with_grad.detach()
        new_actions, log_pi, pre_tanh_value = (
            action_distrib.rsample_logprob_and_pretanh()
        )
        log_pi = log_pi.unsqueeze(1)
        policy_mean = action_distrib.mean
        policy_log_std = torch.log(action_distrib.stddev)

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        unscaled_rewards_flat = rewards.view(t * b, 1)
        rewards_flat = unscaled_rewards_flat * self.reward_scale
        terms_flat = terminals.view(t * b, 1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        if self.backprop_q_loss_into_encoder:
            q1_pred = self.qf1(obs, actions, task_z_with_grad)
            q2_pred = self.qf2(obs, actions, task_z_with_grad)
        else:
            q1_pred = self.qf1(obs, actions, task_z_detached)
            q2_pred = self.qf2(obs, actions, task_z_detached)
        v_pred = self.vf(obs, task_z_detached)
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z_detached)

        """
        QF, Encoder, and Decoder Loss
        """
        # note: encoder/deocder do not get grads from policy or vf
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)

        # KL constraint on z if probabilistic
        kl_div = kl_divergence(p_z, self.agent.latent_prior).sum()
        kl_loss = self.kl_lambda * kl_div
        if self.train_context_decoder:
            # TODO: change to use a distribution
            reward_pred = self.context_decoder(obs, actions, task_z_with_grad)
            reward_prediction_loss = ((reward_pred - unscaled_rewards_flat)**2).mean()
            context_loss = kl_loss + reward_prediction_loss
        else:
            context_loss = kl_loss
            reward_prediction_loss = ptu.zeros(1)

        if self.train_encoder_decoder:
            self.context_optimizer.zero_grad()
        if self.train_agent:
            self.qf1_optimizer.zero_grad()
            self.qf2_optimizer.zero_grad()
        context_loss.backward(retain_graph=True)
        qf_loss.backward()
        if self.train_agent:
            self.qf1_optimizer.step()
            self.qf2_optimizer.step()
        if self.train_encoder_decoder:
            self.context_optimizer.step()

        """
        VF update
        """
        min_q_new_actions = self._min_q(obs, new_actions, task_z_detached)
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        """
        Policy update
        """
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions
        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(p_z.mean)))
                z_sig = np.mean(ptu.get_numpy(p_z.stddev))
                self.eval_statistics['Z mean-abs train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['task_embedding/kl_divergence'] = (
                ptu.get_numpy(kl_div)
            )
            self.eval_statistics['task_embedding/kl_loss'] = (
                ptu.get_numpy(kl_loss)
            )
            self.eval_statistics['task_embedding/reward_prediction_loss'] = (
                ptu.get_numpy(reward_prediction_loss)
            )
            self.eval_statistics['task_embedding/context_loss'] = (
                ptu.get_numpy(context_loss)
            )
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def get_snapshot(self):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            context_decoder=self.context_decoder.state_dict(),
        )
        return snapshot

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats
