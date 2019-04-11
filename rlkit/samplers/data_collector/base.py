import abc


class BaseCollector(object, metaclass=abc.ABCMeta):
    def end_epoch(self, epoch):
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

    @abc.abstractmethod
    def get_epoch_paths(self):
        pass


class PathCollector(BaseCollector, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        pass


class StepCollector(BaseCollector, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        pass
