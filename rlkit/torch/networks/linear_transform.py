from rlkit.torch.core import PyTorchModule


class LinearTransform(PyTorchModule):
    def __init__(self, m, b):
        super().__init__()
        self.m = m
        self.b = b

    def __call__(self, t):
        return self.m * t + self.b
