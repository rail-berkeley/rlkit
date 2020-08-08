from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from rlkit.torch.her.her import HER
from rlkit.demos.td3_bc import TD3BC


class HerTD3BC(HER, TD3BC):
    def __init__(
            self,
            *args,
            td3_kwargs,
            her_kwargs,
            base_kwargs,
            **kwargs
    ):
        HER.__init__(
            self,
            **her_kwargs,
        )
        TD3BC.__init__(self, *args, **kwargs, **td3_kwargs, **base_kwargs)
        assert isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )
