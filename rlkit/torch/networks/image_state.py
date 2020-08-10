from rlkit.policies.base import Policy
from rlkit.torch.core import PyTorchModule, eval_np


class ImageStatePolicy(PyTorchModule, Policy):
    """Switches between image or state inputs"""

    def __init__(
            self,
            image_conv_net,
            state_fc_net,
    ):
        super().__init__()

        assert image_conv_net is None or state_fc_net is None
        self.image_conv_net = image_conv_net
        self.state_fc_net = state_fc_net

    def forward(self, input, return_preactivations=False):
        if self.image_conv_net is not None:
            image = input[:, :21168]
            return self.image_conv_net(image)
        if self.state_fc_net is not None:
            state = input[:, 21168:]
            return self.state_fc_net(state)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class ImageStateQ(PyTorchModule):
    """Switches between image or state inputs"""

    def __init__(
            self,
            # obs_dim,
            # action_dim,
            # goal_dim,
            image_conv_net,  # assumed to be a MergedCNN
            state_fc_net,
    ):
        super().__init__()

        assert image_conv_net is None or state_fc_net is None
        # self.obs_dim = obs_dim
        # self.action_dim = action_dim
        # self.goal_dim = goal_dim
        self.image_conv_net = image_conv_net
        self.state_fc_net = state_fc_net

    def forward(self, input, action, return_preactivations=False):
        if self.image_conv_net is not None:
            image = input[:, :21168]
            return self.image_conv_net(image, action)
        if self.state_fc_net is not None:
            state = input[:, 21168:]  # action + state
            return self.state_fc_net(state, action)


