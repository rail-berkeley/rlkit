import torch

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.networks import MlpPolicy


class AETanhPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """

    def __init__(
            self,
            ae,
            env,
            history_length,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs, output_activation=torch.tanh)
        self.ae = ae
        self.history_length = history_length
        self.env = env

    def get_action(self, obs_np):
        obs = obs_np
        obs = ptu.from_numpy(obs)
        image_obs, fc_obs = self.env.split_obs(obs)
        latent_obs = self.ae.history_encoder(image_obs, self.history_length)
        if fc_obs is not None:
            latent_obs = torch.cat((latent_obs, fc_obs), dim=1)
        obs_np = ptu.get_numpy(latent_obs)[0]
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

