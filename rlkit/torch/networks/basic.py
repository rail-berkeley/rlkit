import torch
from torch import nn


class Clamp(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.__name__ = "Clamp"

    def forward(self, x):
        return torch.clamp(x, **self.kwargs)


class Split(nn.Module):
    """
    Split input and process each chunk with a separate module.
    """
    def __init__(self, module1, module2, split_idx):
        super().__init__()
        self.module1 = module1
        self.module2 = module2
        self.split_idx = split_idx

    def forward(self, x):
        in1 = x[:, :self.split_idx]
        out1 = self.module1(in1)

        in2 = x[:, self.split_idx:]
        out2 = self.module2(in2)

        return out1, out2


class FlattenEach(nn.Module):
    def forward(self, inputs):
        return tuple(x.view(x.size(0), -1) for x in inputs)


class FlattenEachParallel(nn.Module):
    def forward(self, *inputs):
        return tuple(x.view(x.size(0), -1) for x in inputs)


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)


class Map(nn.Module):
    """Apply a module to each input."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return tuple(self.module(x) for x in inputs)


class ApplyMany(nn.Module):
    """Apply many modules to one input."""
    def __init__(self, *modules):
        super().__init__()
        self.modules_to_apply = nn.ModuleList(modules)

    def forward(self, inputs):
        return tuple(m(inputs) for m in self.modules_to_apply)


class LearnedPositiveConstant(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self._constant = nn.Parameter(init_value)

    def forward(self, _):
        return self._constant


class Reshape(nn.Module):
    def __init__(self, *output_shape):
        super().__init__()
        self._output_shape_with_batch_size = (-1, *output_shape)

    def forward(self, inputs):
        return inputs.view(self._output_shape_with_batch_size)


class ConcatTuple(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


class MultiInputSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input


class Detach(nn.Module):
    def __init__(self, wrapped_mlp):
        super().__init__()
        self.wrapped_mlp = wrapped_mlp

    def forward(self, inputs):
        return self.wrapped_mlp.forward(inputs).detach()

    def __getattr__(self, attr_name):
        try:
            return super().__getattr__(attr_name)
        except AttributeError:
            return getattr(self.wrapped_mlp, attr_name)
