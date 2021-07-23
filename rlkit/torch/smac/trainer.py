from collections import OrderedDict
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn as nn
from torch.distributions import kl_divergence

import rlkit.torch.pytorch_util as ptu
from rlkit.core.logging import add_prefix
from rlkit.util import ml_util
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.networks import LinearTransform
from rlkit.torch.smac.agent import SmacAgent
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class SmacTrainer(TorchTrainer):
    def __init__(
            self,
            agent: SmacAgent,
            env,
            latent_dim,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            context_encoder,
            reward_predictor,
            context_decoder,

            train_context_decoder=False,
            backprop_q_loss_into_encoder=True,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,

            train_reward_pred_in_unsupervised_phase=False,
            use_encoder_snapshot_for_reward_pred_in_unsupervised_phase=False,

            # from AWAC
            buffer_policy=None,

            discount=0.99,
            reward_scale=1.0,
            beta=1.0,
            beta_schedule_kwargs=None,

            policy_lr=1e-3,
            qf_lr=1e-3,
            policy_weight_decay=0,
            q_weight_decay=0,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,

            bc_num_pretrain_steps=0,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=0,
            bc_batch_size=128,
            alpha=1.0,

            policy_update_period=1,
            q_update_period=1,

            weight_loss=True,
            compute_bc=True,
            use_awr_update=True,
            use_reparam_update=False,

            bc_weight=0.0,
            rl_weight=1.0,
            reparam_weight=1.0,
            reparam_weight_schedule_kwargs=None,
            awr_weight=1.0,

            awr_use_mle_for_vf=False,
            vf_K=1,
            awr_sample_actions=False,
            buffer_policy_sample_actions=False,
            awr_min_q=False,
            brac=False,

            reward_transform_class=None,
            reward_transform_kwargs=None,
            terminal_transform_class=None,
            terminal_transform_kwargs=None,

            pretraining_logging_period=1000,

            train_bc_on_rl_buffer=False,
            use_automatic_beta_tuning=False,
            beta_epsilon=1e-10,
            normalize_over_batch=True,
            Z_K=10,
            clip_score=None,
            validation_qlearning=False,

            mask_positive_advantage=False,
            buffer_policy_reset_period=-1,
            num_buffer_policy_train_steps_on_reset=100,
            advantage_weighted_buffer_loss=True,

            # for debugging
            _debug_ignore_context=False,
            _debug_use_ground_truth_context=False,
    ):
        super().__init__()

        self.train_agent = True
        self.train_context_decoder = train_context_decoder
        self.train_encoder_decoder = True
        self.backprop_q_loss_into_encoder = backprop_q_loss_into_encoder
        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
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

        self._debug_ignore_context = _debug_ignore_context

        self.agent = agent
        self.policy = agent.policy
        self.qf1, self.qf2 = qf1, qf2
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

        self.eval_statistics = None
        self._need_to_update_eval_statistics = True

        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.buffer_policy = buffer_policy
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_awr_update = use_awr_update
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.awr_use_mle_for_vf = awr_use_mle_for_vf
        self.vf_K = vf_K
        self.awr_sample_actions = awr_sample_actions
        self.awr_min_q = awr_min_q

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()

        self.optimizers = {}

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            weight_decay=policy_weight_decay,
            lr=policy_lr,
        )
        self.optimizers[self.policy] = self.policy_optimizer
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )

        self.use_automatic_beta_tuning = use_automatic_beta_tuning and buffer_policy and train_bc_on_rl_buffer
        self.beta_epsilon = beta_epsilon
        if self.use_automatic_beta_tuning:
            self.log_beta = ptu.zeros(1, requires_grad=True)
            self.beta_optimizer = optimizer_class(
                [self.log_beta],
                lr=policy_lr,
            )
        else:
            self.beta = beta
            self.beta_schedule_kwargs = beta_schedule_kwargs
            if beta_schedule_kwargs is None:
                self.beta_schedule = ml_util.ConstantSchedule(beta)
            else:
                schedule_class = beta_schedule_kwargs.pop("schedule_class",
                                                          ml_util.PiecewiseLinearSchedule)
                self.beta_schedule = schedule_class(**beta_schedule_kwargs)

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.bc_num_pretrain_steps = bc_num_pretrain_steps
        self.q_num_pretrain1_steps = q_num_pretrain1_steps
        self.q_num_pretrain2_steps = q_num_pretrain2_steps
        self.bc_batch_size = bc_batch_size
        self.rl_weight = rl_weight
        self.bc_weight = bc_weight
        self.compute_bc = compute_bc
        self.alpha = alpha
        self.q_update_period = q_update_period
        self.policy_update_period = policy_update_period
        self.weight_loss = weight_loss

        self.reparam_weight = reparam_weight
        self.reparam_weight_schedule = None
        self.reparam_weight_schedule_kwargs = reparam_weight_schedule_kwargs
        self.awr_weight = awr_weight
        self.update_policy = True
        self.pretraining_logging_period = pretraining_logging_period
        self.normalize_over_batch = normalize_over_batch
        self.Z_K = Z_K

        self.reward_transform_class = reward_transform_class or LinearTransform
        self.reward_transform_kwargs = reward_transform_kwargs or dict(m=1, b=0)
        self.terminal_transform_class = terminal_transform_class or LinearTransform
        self.terminal_transform_kwargs = terminal_transform_kwargs or dict(m=1,
                                                                           b=0)
        self.reward_transform = self.reward_transform_class(
            **self.reward_transform_kwargs)
        self.terminal_transform = self.terminal_transform_class(
            **self.terminal_transform_kwargs)
        self.use_reparam_update = use_reparam_update
        self.clip_score = clip_score
        self.buffer_policy_sample_actions = buffer_policy_sample_actions

        self.train_bc_on_rl_buffer = train_bc_on_rl_buffer and buffer_policy
        self.validation_qlearning = validation_qlearning
        self.brac = brac
        self.mask_positive_advantage = mask_positive_advantage
        self.buffer_policy_reset_period = buffer_policy_reset_period
        self.num_buffer_policy_train_steps_on_reset = num_buffer_policy_train_steps_on_reset
        self.advantage_weighted_buffer_loss = advantage_weighted_buffer_loss
        self._debug_use_ground_truth_context = _debug_use_ground_truth_context
        self._num_gradient_steps = 0

    @property
    def train_reparam_weight(self):
        if self.reparam_weight_schedule_kwargs is not None and self.reparam_weight_schedule is None:
            self.reparam_weight_schedule = ml_util.create_schedule(
                **self.reparam_weight_schedule_kwargs
            )
        if self.reparam_weight_schedule is None:
            return self.reparam_weight
        else:
            return self.reparam_weight_schedule.get_value(
                self._n_train_steps_total
            )

    ##### Training #####
    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        context = batch['context']

        if self.reward_transform:
            rewards = self.reward_transform(rewards)

        if self.terminal_transform:
            terminals = self.terminal_transform(terminals)
        """
        Policy and Alpha Loss
        """
        dist, p_z, task_z_with_grad = self.agent(
            obs, context, return_latent_posterior_and_task_z=True,
        )
        task_z_detached = task_z_with_grad.detach()
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(1)
        next_dist = self.agent(next_obs, context)

        if self._debug_ignore_context:
            task_z_with_grad = task_z_with_grad * 0

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        unscaled_rewards_flat = rewards.view(t * b, 1)
        rewards_flat = unscaled_rewards_flat * self.reward_scale
        terms_flat = terminals.view(t * b, 1)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (
                        log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha

        """
        QF Loss
        """
        if self.backprop_q_loss_into_encoder:
            q1_pred = self.qf1(obs, actions, task_z_with_grad)
            q2_pred = self.qf2(obs, actions, task_z_with_grad)
        else:
            q1_pred = self.qf1(obs, actions, task_z_detached)
            q2_pred = self.qf2(obs, actions, task_z_detached)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(1)
        with torch.no_grad():
            target_q_values = torch.min(
                self.target_qf1(next_obs, new_next_actions, task_z_detached),
                self.target_qf2(next_obs, new_next_actions, task_z_detached),
            ) - alpha * new_log_pi

        q_target = rewards_flat + (
                    1. - terms_flat) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Context Encoder Loss
        """
        if self._debug_use_ground_truth_context:
            kl_div = kl_loss = ptu.zeros(0)
        else:
            kl_div = kl_divergence(p_z, self.agent.latent_prior).mean(dim=0).sum()
            kl_loss = self.kl_lambda * kl_div

        if self.train_context_decoder:
            # TODO: change to use a distribution
            reward_pred = self.context_decoder(obs, actions, task_z_with_grad)
            reward_prediction_loss = ((reward_pred - unscaled_rewards_flat)**2).mean()
            context_loss = kl_loss + reward_prediction_loss
        else:
            context_loss = kl_loss
            reward_prediction_loss = ptu.zeros(1)

        """
        Policy Loss
        """
        qf1_new_actions = self.qf1(obs, new_obs_actions, task_z_detached)
        qf2_new_actions = self.qf2(obs, new_obs_actions, task_z_detached)
        q_new_actions = torch.min(
            qf1_new_actions,
            qf2_new_actions,
        )

        # Advantage-weighted regression
        if self.vf_K > 1:
            vs = []
            for i in range(self.vf_K):
                u = dist.sample()
                q1 = self.qf1(obs, u, task_z_detached)
                q2 = self.qf2(obs, u, task_z_detached)
                v = torch.min(q1, q2)
                # v = q1
                vs.append(v)
            v_pi = torch.cat(vs, 1).mean(dim=1)
        else:
            # v_pi = self.qf1(obs, new_obs_actions)
            v1_pi = self.qf1(obs, new_obs_actions, task_z_detached)
            v2_pi = self.qf2(obs, new_obs_actions, task_z_detached)
            v_pi = torch.min(v1_pi, v2_pi)

        u = actions
        if self.awr_min_q:
            q_adv = torch.min(q1_pred, q2_pred)
        else:
            q_adv = q1_pred

        policy_logpp = dist.log_prob(u)

        if self.use_automatic_beta_tuning:
            buffer_dist = self.buffer_policy(obs)
            beta = self.log_beta.exp()
            kldiv = torch.distributions.kl.kl_divergence(dist, buffer_dist)
            beta_loss = -1 * (
                        beta * (kldiv - self.beta_epsilon).detach()).mean()

            self.beta_optimizer.zero_grad()
            beta_loss.backward()
            self.beta_optimizer.step()
        else:
            beta = self.beta_schedule.get_value(self._n_train_steps_total)
            beta_loss = ptu.zeros(1)

        score = q_adv - v_pi
        if self.mask_positive_advantage:
            score = torch.sign(score)

        if self.clip_score is not None:
            score = torch.clamp(score, max=self.clip_score)

        weights = batch.get('weights', None)
        if self.weight_loss and weights is None:
            if self.normalize_over_batch == True:
                weights = F.softmax(score / beta, dim=0)
            elif self.normalize_over_batch == "whiten":
                adv_mean = torch.mean(score)
                adv_std = torch.std(score) + 1e-5
                normalized_score = (score - adv_mean) / adv_std
                weights = torch.exp(normalized_score / beta)
            elif self.normalize_over_batch == "exp":
                weights = torch.exp(score / beta)
            elif self.normalize_over_batch == "step_fn":
                weights = (score > 0).float()
            elif self.normalize_over_batch == False:
                weights = score
            elif self.normalize_over_batch == 'uniform':
                weights = F.softmax(ptu.ones_like(score) / beta, dim=0)
            else:
                raise ValueError(self.normalize_over_batch)
        weights = weights[:, 0]

        policy_loss = alpha * log_pi.mean()

        if self.use_awr_update and self.weight_loss:
            policy_loss = policy_loss + self.awr_weight * (
                        -policy_logpp * len(weights) * weights.detach()).mean()
        elif self.use_awr_update:
            policy_loss = policy_loss + self.awr_weight * (-policy_logpp).mean()

        if self.use_reparam_update:
            policy_loss = policy_loss + self.train_reparam_weight * (
                -q_new_actions).mean()

        policy_loss = self.rl_weight * policy_loss

        """
        Update networks
        """
        if self._n_train_steps_total % self.q_update_period == 0:
            if self.train_encoder_decoder:
                self.context_optimizer.zero_grad()
            if self.train_agent:
                self.qf1_optimizer.zero_grad()
                self.qf2_optimizer.zero_grad()
            context_loss.backward(retain_graph=True)
            # retain graph because the encoder is trained by both QF losses
            qf1_loss.backward(retain_graph=True)
            qf2_loss.backward()
            if self.train_agent:
                self.qf1_optimizer.step()
                self.qf2_optimizer.step()
            if self.train_encoder_decoder:
                self.context_optimizer.step()

        if self.train_agent:
            if self._n_train_steps_total % self.policy_update_period == 0 and self.update_policy:
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
        self._num_gradient_steps += 1

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
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
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'rewards',
                ptu.get_numpy(rewards),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'terminals',
                ptu.get_numpy(terminals),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            self.eval_statistics.update(policy_statistics)
            self.eval_statistics.update(create_stats_ordered_dict(
                'Advantage Weights',
                ptu.get_numpy(weights),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Advantage Score',
                ptu.get_numpy(score),
            ))
            self.eval_statistics['reparam_weight'] = self.train_reparam_weight
            self.eval_statistics['num_gradient_steps'] = (
                self._num_gradient_steps
            )

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

            if self.use_automatic_beta_tuning:
                self.eval_statistics.update({
                    "adaptive_beta/beta": ptu.get_numpy(beta.mean()),
                    "adaptive_beta/beta loss": ptu.get_numpy(beta_loss.mean()),
                })

        self._n_train_steps_total += 1

    def configure(self, **params):
        for k, v in params.items():
            if k not in self.__dict__:
                raise KeyError('Member {} is not in {}'.format(k, self))
            self.__dict__[k] = v

    #### Trainer ####
    def get_snapshot(self):
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            target_qf1=self.target_qf1.state_dict(),
            target_qf2=self.target_qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
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

    ###### Torch stuff #####
    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
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
