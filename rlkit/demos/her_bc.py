from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from rlkit.torch.her.her import HER
from rlkit.demos.behavior_clone import BehaviorClone


class HerBC(HER, BehaviorClone):
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
        BehaviorClone.__init__(self, *args, **kwargs, **td3_kwargs, **base_kwargs)
        assert isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )

    def _handle_rollout_ending(self):
        """Don't add anything to rollout buffer"""
        self._n_rollouts_total += 1
        # if len(self._current_path_builder) > 0:
            # path = self._current_path_builder.get_all_stacked()
            # self.replay_buffer.add_path(path)
            # self._exploration_paths.append(path)
            # self._current_path_builder = PathBuilder()

    def _handle_path(self, path):
        """Don't add anything to rollout buffer"""
        self._n_rollouts_total += 1
        # self.replay_buffer.add_path(path)
        self._exploration_paths.append(path)
