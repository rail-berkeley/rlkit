from collections import OrderedDict
import numpy as np
import pickle

import torch
import torch.optim as optim
from torch import nn as nn
from torch.autograd import Variable

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_ify, torch_ify
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import MetaTorchRLAlgorithm


def product_of_gaussians(mus, sigmas):
    """
    :param mus: Tensor containing means
    :param sigmas: Tensor containing sigmas
    :return: Tensor containing mu, sigma of the Gaussian likelihood formed by the product of Gaussian likelihoods
    """
    sigmas_squared = sigmas ** 2
    sigma_squared = 1. / torch.sum(torch.reciprocol(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, torch.sqrt(sigma_squared)

def mean_of_gaussians(mus, sigmas):
    """
    :param mus: Tensor containing means
    :param sigmas: Tensor containing sigmas
    :return: Tensor containing mu, sigma of the Gaussian likelihood formed by the means of the Gaussians
    """
    mu = torch.mean(mus, dim=0)
    sigma = torch.sqrt(torch.mean(sigmas**2, dim=0))
    return mu, sigma

class ProtoSoftActorCritic(MetaTorchRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            class_lr=1e-1,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            reparameterize=True,
            use_information_bottleneck=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            eval_deterministic=True, # TODO: use this flag in evals
            **kwargs
    ):
        self.task_enc, self.policy, self.qf1, self.qf2, self.vf, self.rf = nets
        super().__init__(
            env=env,
            policy=self.policy,
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.target_vf = self.vf.copy()
        self.class_criterion = nn.BCEWithLogitsLoss()
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.rf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.eval_statistics = None
        self.latent_dim = latent_dim

        self.reparameterize = reparameterize
        self.use_information_bottleneck = use_information_bottleneck

        self.class_optimizer = optim.SGD(
                self.task_enc.parameters(),
                lr=class_lr,
        )

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
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
        self.context_optimizer = optimizer_class(
            self.task_enc.parameters(),
            lr=context_lr,
        )
        self.rf_optimizer = optimizer_class(
            self.rf.parameters(),
            lr=context_lr,
        )

    # TODO: leave for now?
    def dense_to_sparse(self, rewards):
        # rewards_np = rewards.data.numpy()
        # sparse_reward = (rewards_np < .2).astype(int)
        # return torch.autograd.Variable(torch.FloatTensor(sparse_reward), requires_grad=False)
        return rewards

    def perform_meta_update(self):
        self.context_optimizer.step()
        self.rf_optimizer.step()
        self._update_target_network()

        self.rf_optimizer.zero_grad()
        self.context_optimizer.zero_grad()

    # TODO: implement this
    def _do_information_bottleneck(self):

        batch_size = 256
        batches = []
        for idx in self.train_tasks:
            batch = self.get_encoding_batch(eval_task=False)
            rewards = self.dense_to_sparse(batch['rewards'])
            terminals = batch['terminals']
            obs = batch['observations']
            actions = batch['actions']
            next_obs = batch['next_observations']

        # mu, sigma = self.product_of_gaussians(batch)

        z_dist = torch.distributions.Normal(mu, sigma)
        kl_loss = torch.sum(torch.distributions.kl.kl_divergence(
            z_dist, torch.distributions.Normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))))
        kl_loss.backward(retain_graph=True)

    def compute_embedding_from_batch(self, batch, deterministic_embedding=False):
        """
        Computes the embedding for a batch of data
        :param batch: dictionary of tensors, see get_encoding_batch
        :param deterministic_embedding: If we have a Gaussian distribution over z, this flag will select the mean
               instaed of sampling
        :return: Tensor of shape [latent_dim]
        """
        rewards = batch['rewards']
        obs = batch['observations']

        enc_outputs = self.task_enc(obs, self.dense_to_sparse(rewards) / self.reward_scale)

        if not self.use_information_bottleneck:
            return torch.mean(enc_outputs, dim=0)

        mus = enc_outputs[:, :self.latent_dim]
        sigmas = torch.nn.Softplus(enc_outputs[:, self.latent_dim:])

        mu, sigma = mean_of_gaussians(mus, sigmas)

        if deterministic_embedding:
            return mu

        z_dist = torch.Distributions.Normal(mu, sigma)
        return z_dist.sample()

    def _do_training(self, idx, epoch):
        # sample from replay buffer to compute training update
        batch = self.get_batch(idx)
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        batch_enc = self.get_encoding_batch(eval_task=False, idx=idx)
        obs_enc = batch_enc['observations']
        rewards_enc = batch_enc['rewards']

        # Product of Gaussians.
        z = self.compute_embedding_from_batch(batch_enc)
        batch_z_enc = z.repeat(obs_enc.shape[0], 1)

        r_pred = self.rf(obs_enc, batch_z_enc)
        rf_loss = 1. * self.rf_criterion(r_pred, rewards_enc)
        rf_loss.backward(retain_graph=True)

        batch_z = z.repeat(obs.shape[0], 1)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        q1_pred = self.qf1(obs, actions, batch_z)
        q2_pred = self.qf2(obs, actions, batch_z)
        v_pred = self.vf(obs, batch_z.detach())
        # make sure policy accounts for squashing functions like tanh correctly!
        in_ = torch.cat([obs, batch_z.detach()], dim=1)
        policy_outputs = self.policy(in_, reparameterize=self.reparameterize, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # qf loss and gradients
        # do residual q next
        target_v_values = self.target_vf(next_obs, batch_z.detach())
        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        # no detach here for residual gradient through batch_z
        qf_loss = torch.mean((q1_pred - q_target) ** 2)
        qf_loss += torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        # vf loss and gradients
        self.vf_optimizer.zero_grad()
        q1_new_actions = self.qf1(obs, new_actions, batch_z.detach())
        q2_new_actions = self.qf2(obs, new_actions, batch_z.detach())
        min_q_new_actions = torch.min(q1_new_actions, q2_new_actions)
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        vf_loss.backward(retain_graph=True)
        self.vf_optimizer.step()

        # policy loss and gradients
        self.policy_optimizer.zero_grad()
        log_policy_target = q1_new_actions

        if self.reparameterize:
            policy_loss = (
                    log_pi - log_policy_target
            ).mean()
        else:
            policy_loss = (
                log_pi * (log_pi - log_policy_target + v_pred).detach()
            ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['RF Loss'] = np.mean(ptu.get_numpy(rf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'R Predictions',
                ptu.get_numpy(r_pred)
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

    def sample_z_from_prior(self):
        if self.use_information_bottleneck:
            return np.random.normal(size=self.latent_dim)
        else:
            return np.zeros(self.latent_dim)

    def sample_z_from_posterior(self, idx, eval_task=False):
        batch = self.get_encoding_batch(idx=idx, eval_task=eval_task)

        # replace with generic compute embedding from batch function
        z = self.compute_embedding_from_batch(batch)
        return np_ify(z)

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.vf,
            self.rf,
            self.target_vf,
            self.task_enc,
        ]

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.policy,
            vf=self.vf,
            rf=self.rf,
            target_vf=self.target_vf,
            task_enc=self.task_enc,
        )
        return snapshot
