"""Code based on https://github.com/katerakelly/oyster"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from rlkit.util.wrapper import Wrapper
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import Policy
from rlkit.torch.distributions import (
    Delta,
)
from rlkit.torch.sac.policies import MakeDeterministic


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class SmacAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 policy,
                 reward_predictor,
                 use_next_obs_in_context=False,
                 _debug_ignore_context=False,
                 _debug_do_not_sqrt=False,
                 _debug_use_ground_truth_context=False
                 ):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder
        self.policy = policy
        self.reward_predictor = reward_predictor
        self.deterministic_policy = MakeDeterministic(self.policy)
        self._debug_ignore_context = _debug_ignore_context
        self._debug_use_ground_truth_context = _debug_use_ground_truth_context

        # self.recurrent = kwargs['recurrent']
        # self.use_ib = kwargs['use_information_bottleneck']
        # self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = use_next_obs_in_context

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.z_means = None
        self.z_vars = None
        self.context = None
        self.z = None

        # rp = reward predictor
        # TODO: add back in reward predictor code
        self.z_means_rp = None
        self.z_vars_rp = None
        self.z_rp = None
        self.context_encoder_rp = context_encoder
        self._use_context_encoder_snapshot_for_reward_pred = False

        self.latent_prior = torch.distributions.Normal(
            ptu.zeros(self.latent_dim),
            ptu.ones(self.latent_dim)
        )

        self._debug_do_not_sqrt = _debug_do_not_sqrt

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        #  reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        var = ptu.ones(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var

    @property
    def use_context_encoder_snapshot_for_reward_pred(self):
        return self._use_context_encoder_snapshot_for_reward_pred

    @use_context_encoder_snapshot_for_reward_pred.setter
    def use_context_encoder_snapshot_for_reward_pred(self, value):
        if value and not self.use_context_encoder_snapshot_for_reward_pred:
            # copy context encoder on switch
            self.context_encoder_rp = copy.deepcopy(self.context_encoder)
            self.context_encoder_rp.to(ptu.device)
            self.reward_predictor = copy.deepcopy(self.reward_predictor)
            self.reward_predictor.to(ptu.device)
        self._use_context_encoder_snapshot_for_reward_pred = value

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

        self.z_rp = self.z_rp.detach()
        if self.recurrent:
            self.context_encoder_rp.hidden = self.context_encoder_rp.hidden.detach()

    def update_context(self, context, inputs):
        ''' append single transition to the current context '''
        if self._debug_use_ground_truth_context:
            return context
        o, a, r, no, d, info = inputs
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])

        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if context is None:
            context = data
        else:
            try:
                context = torch.cat([context, data], dim=1)
            except Exception as e:
                import ipdb; ipdb.set_trace()
        return context

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def batched_latent_prior(self, batch_size):
        return torch.distributions.Normal(
            ptu.zeros(batch_size, self.latent_dim),
            ptu.ones(batch_size, self.latent_dim)
        )

    def latent_posterior(self, context, squeeze=False, for_reward_prediction=False):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        if isinstance(context, np.ndarray):
            context = ptu.from_numpy(context)
        if self._debug_use_ground_truth_context:
            if squeeze:
                context = context.squeeze(dim=0)
            return Delta(context)
        if for_reward_prediction:
            context_encoder = self.context_encoder_rp
        else:
            context_encoder = self.context_encoder
        params = context_encoder(context)
        params = params.view(context.size(0), -1, context_encoder.output_size)
        mu = params[..., :self.latent_dim]
        sigma_squared = F.softplus(params[..., self.latent_dim:])
        z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        z_means = torch.stack([p[0] for p in z_params])
        z_vars = torch.stack([p[1] for p in z_params])
        if squeeze:
            z_means = z_means.squeeze(dim=0)
            z_vars = z_vars.squeeze(dim=0)
        if self._debug_do_not_sqrt:
            return torch.distributions.Normal(z_means, z_vars)
        else:
            return torch.distributions.Normal(z_means, torch.sqrt(z_vars))

    def get_action(self, obs, z, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        obs = ptu.from_numpy(obs[None])
        if self._debug_ignore_context:
            z = ptu.from_numpy(z[None]) * 0
        else:
            z = ptu.from_numpy(z[None])
        if len(obs.shape) != len(z.shape):
            import ipdb; ipdb.set_trace()
        in_ = torch.cat([obs, z], dim=1)[0]
        if deterministic:
            return self.deterministic_policy.get_action(in_)
        else:
            return self.policy.get_action(in_)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(
            self, obs, context,
            return_task_z=False,
            return_latent_posterior=False,
            return_latent_posterior_and_task_z=False,
    ):
        ''' given context, get statistics under the current policy of a set of observations '''
        context_distrib = self.latent_posterior(context)
        task_z = context_distrib.rsample()

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        action_distribution = self.policy(in_)
        # policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)
        if return_latent_posterior_and_task_z:
            return action_distribution, context_distrib, task_z
        if return_latent_posterior:
            return action_distribution, context_distrib
        if return_task_z:
            return action_distribution, task_z
        else:
            return action_distribution

        # return policy_outputs, task_z

    def infer_reward(self, obs, action, z):
        obs = ptu.from_numpy(obs[None])
        action = ptu.from_numpy(action[None])
        z = ptu.from_numpy(z[None])
        reward = self.reward_predictor(obs, action, z)
        return ptu.get_numpy(reward)[0]

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

        # z_mean_rp = np.mean(np.abs(ptu.get_numpy(self.z_means_rp[0])))
        # z_sig_rp = np.mean(ptu.get_numpy(self.z_vars_rp[0]))
        # eval_statistics['Z rew-pred mean eval'] = z_mean_rp
        # eval_statistics['Z rew-pred variance eval'] = z_sig_rp

    @property
    def networks(self):
        if self.context_encoder is self.context_encoder_rp:
            return [self.context_encoder, self.policy]
        else:
            return [self.context_encoder, self.context_encoder_rp, self.policy]


class MakeSMACAgentDeterministic(Wrapper, Policy):
    def __init__(self, stochastic_policy):
        super().__init__(stochastic_policy)
        self.stochastic_policy = stochastic_policy

    def get_action(self, *args):
        return self.stochastic_policy.get_action(*args, deterministic=True)

    def get_actions(self, *args):
        return self.stochastic_policy.get_actions(*args, deterministic=True)
