import abc
import numpy as np
from collections import OrderedDict

from torch import nn as nn
from torch.autograd import Variable

from rlkit.torch import pytorch_util as ptu
from rlkit.core.serializable import Serializable


class PyTorchModule(nn.Module, Serializable, metaclass=abc.ABCMeta):

    def get_param_values(self):
        return self.state_dict()

    def set_param_values(self, param_values):
        self.load_state_dict(param_values)

    def get_param_values_np(self):
        state_dict = self.state_dict()
        np_dict = OrderedDict()
        for key, tensor in state_dict.items():
            np_dict[key] = ptu.get_numpy(tensor)
        return np_dict

    def set_param_values_np(self, param_values):
        torch_dict = OrderedDict()
        for key, tensor in param_values.items():
            torch_dict[key] = ptu.from_numpy(tensor)
        self.load_state_dict(torch_dict)

    def copy(self):
        copy = Serializable.clone(self)
        ptu.copy_model_params_from_to(self, copy)
        return copy

    def save_init_params(self, locals):
        """
        Should call this FIRST THING in the __init__ method if you ever want
        to serialize or clone this network.

        Usage:
        ```
        def __init__(self, ...):
            self.init_serialization(locals())
            ...
        ```
        :param locals:
        :return:
        """
        Serializable.quick_init(self, locals)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.set_param_values(d["params"])

    def regularizable_parameters(self):
        """
        Return generator of regularizable parameters. Right now, all non-flat
        vectors are assumed to be regularizabled, presumably because only
        biases are flat.

        :return:
        """
        for param in self.parameters():
            if len(param.size()) > 1:
                yield param

    def eval_np(self, *args, **kwargs):
        """
        Eval this module with a numpy interface

        Same as a call to __call__ except all Variable input/outputs are
        replaced with numpy equivalents.

        Assumes the output is either a single object or a tuple of objects.
        """
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        outputs = self.__call__(*torch_args, **torch_kwargs)
        if isinstance(outputs, tuple):
            return tuple(np_ify(x) for x in outputs)
        else:
            return np_ify(outputs)


def torch_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return ptu.from_numpy(np_array_or_other)
    else:
        return np_array_or_other


def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, Variable):
        return ptu.get_numpy(tensor_or_other)
    else:
        return tensor_or_other
