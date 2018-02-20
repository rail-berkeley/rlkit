import numpy as np
import torch

from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.networks import TanhMlpPolicy, FlattenMlp


class TdmNormalizer(object):
    def __init__(
            self,
            env,
            vectorized,
            normalize_tau=False,
            max_tau=0,
            log_tau=False,
    ):
        if normalize_tau:
            assert max_tau > 0, "Max tau must be larger than 0 if normalizing"
        self.observation_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.goal_dim = env.goal_dim
        self.obs_normalizer = TorchFixedNormalizer(self.observation_dim)
        self.goal_normalizer = TorchFixedNormalizer(env.goal_dim)
        self.action_normalizer = TorchFixedNormalizer(self.action_dim)
        self.distance_normalizer = TorchFixedNormalizer(
            env.goal_dim if vectorized else 1
        )
        self.log_tau = log_tau
        self.normalize_tau = normalize_tau
        self.max_tau = max_tau

        # Assuming that the taus are sampled uniformly from [0, max_tau]
        if self.log_tau:
            # If max_tau = 1, then
            # mean = \int_2^3 log(x) dx ~ 0.9095...
            # std = sqrt{  \int_2^3 (log(x) - mean)^2 dx    } ~ 0.165...
            # Thanks wolfram!
            self.tau_mean = self.max_tau * 0.90954250488443855
            self.tau_std = self.max_tau * 0.11656876357329767
        else:
            self.tau_mean = self.max_tau / 2
            self.tau_std = self.max_tau / np.sqrt(12)

    def normalize_num_steps_left(self, num_steps_left):
        if self.log_tau:
            # minimum tau is -1 (although the output should be ignored for
            # the `tau == -1` case.
            num_steps_left = torch.log(num_steps_left + 2)
        if self.normalize_tau:
            num_steps_left = (num_steps_left - self.tau_mean) / self.tau_std
        return num_steps_left

    def normalize_all(
            self,
            obs,
            actions,
            goals,
            num_steps_left
    ):
        if obs is not None:
            obs = self.obs_normalizer.normalize(obs)
        if actions is not None:
            actions = self.action_normalizer.normalize(actions)
        if goals is not None:
            goals = self.goal_normalizer.normalize(goals)
        if num_steps_left is not None:
            num_steps_left = self.normalize_num_steps_left(num_steps_left)
        return obs, actions, goals, num_steps_left

    def copy_stats(self, other):
        self.obs_normalizer.copy_stats(other.obs_normalizer)
        self.goal_normalizer.copy_stats(other.goal_normalizer)
        self.action_normalizer.copy_stats(other.action_normalizer)
        self.distance_normalizer.copy_stats(other.distance_normalizer)


class TdmQf(FlattenMlp):
    def __init__(
            self,
            env,
            vectorized,
            norm_order,
            structure='norm_difference',
            tdm_normalizer: TdmNormalizer=None,
            **kwargs
    ):
        """

        :param env:
        :param hidden_sizes:
        :param vectorized: Boolean. Vectorized or not?
        :param norm_order: int, 1 or 2. What L norm to use.
        :param structure: String defining output structure of network:
            - 'norm_difference': Q = -||g - f(inputs)||
            - 'none': Q = f(inputs)

        :param kwargs:
        """
        assert structure in [
            'norm_difference',
            'none',
        ]
        if structure == 'difference':
            assert vectorized, "difference only makes sense for vectorized"
        self.save_init_params(locals())
        self.observation_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.goal_dim = env.goal_dim
        super().__init__(
            input_size=(
                    self.observation_dim + self.action_dim + self.goal_dim + 1
            ),
            output_size=self.goal_dim if vectorized else 1,
            **kwargs
        )
        self.env = env
        self.vectorized = vectorized
        self.norm_order = norm_order
        self.structure = structure
        self.tdm_normalizer = tdm_normalizer

    def forward(
            self,
            observations,
            actions,
            goals,
            num_steps_left,
            return_internal_prediction=False,
    ):
        if self.tdm_normalizer is not None:
            observations, actions, goals, num_steps_left = (
                self.tdm_normalizer.normalize_all(
                    observations, actions, goals, num_steps_left
                )
            )

        predictions = super().forward(
            observations, actions, goals, num_steps_left
        )
        if return_internal_prediction:
            return predictions

        if self.vectorized:
            if self.structure == 'norm_difference':
                output = - torch.abs(goals - predictions)
            elif self.structure == 'none':
                output = predictions
            else:
                raise TypeError("Invalid structure: {}".format(self.structure))
        else:
            if self.structure == 'norm_difference':
                output = - torch.norm(
                    goals - predictions,
                    p=self.norm_order,
                    dim=1,
                    keepdim=True,
                )
            elif self.structure == 'none':
                output = predictions
            else:
                raise TypeError(
                    "For vectorized={0}, invalid structure: {1}".format(
                        self.vectorized,
                        self.structure,
                    )
                )
        if self.tdm_normalizer is not None:
            output = self.tdm_normalizer.distance_normalizer.denormalize_scale(
                output
            )
        return output


class TdmPolicy(TanhMlpPolicy):
    """
    Rather than giving `g`, give `g - goalify(s)` as input.
    """
    def __init__(
            self,
            env,
            tdm_normalizer: TdmNormalizer=None,
            **kwargs
    ):
        self.save_init_params(locals())
        self.observation_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.goal_dim = env.goal_dim
        super().__init__(
            input_size=self.observation_dim + self.goal_dim + 1,
            output_size=self.action_dim,
            **kwargs
        )
        self.env = env
        self.tdm_normalizer = tdm_normalizer

    def forward(
            self,
            observations,
            goals,
            num_steps_left,
            return_preactivations=False,
    ):
        if self.tdm_normalizer is not None:
            observations, _, goals, num_steps_left = (
                self.tdm_normalizer.normalize_all(
                    observations, None, goals, num_steps_left
                )
            )

        flat_input = torch.cat((observations, goals, num_steps_left), dim=1)
        return super().forward(
            flat_input,
            return_preactivations=return_preactivations,
        )

    def get_action(self, ob_np, goal_np, tau_np):
        actions = self.eval_np(
            ob_np[None],
            goal_np[None],
            tau_np[None],
        )
        return actions[0, :], {}

