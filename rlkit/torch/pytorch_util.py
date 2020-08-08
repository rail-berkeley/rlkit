import torch
import numpy as np
from torch import nn


def identity(x):
    return x


_str_to_activation = {
    'identity': identity,
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
}


def activation_from_string(string):
    return _str_to_activation[string]


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def maximum_2d(t1, t2):
    # noinspection PyArgumentList
    return torch.max(
        torch.cat((t1.unsqueeze(2), t2.unsqueeze(2)), dim=2),
        dim=2,
    )[0].squeeze(2)


def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    # TODO(vitchyr): see if you can use expand instead of repeat
    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
            .unsqueeze(3)
            .repeat(1, t2_height, t2_width, 1)
            .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2


def alpha_dropout(
        x,
        p=0.05,
        alpha=-1.7580993408473766,
        fixedPointMean=0,
        fixedPointVar=1,
        training=False,
):
    keep_prob = 1 - p
    if keep_prob == 1 or not training:
        return x
    a = np.sqrt(fixedPointVar / (keep_prob * (
            (1 - keep_prob) * pow(alpha - fixedPointMean, 2) + fixedPointVar)))
    b = fixedPointMean - a * (
            keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
    keep_prob = 1 - p

    random_tensor = keep_prob + torch.rand(x.size())
    binary_tensor = torch.floor(random_tensor)
    x = x.mul(binary_tensor)
    ret = x + alpha * (1 - binary_tensor)
    ret.mul_(a).add_(b)
    return ret


def alpha_selu(x, training=False):
    return alpha_dropout(nn.SELU(x), training=training)


def double_moments(x, y):
    """
    Returns the first two moments between x and y.

    Specifically, for each vector x_i and y_i in x and y, compute their
    outer-product. Flatten this resulting matrix and return it.

    The first moments (i.e. x_i and y_i) are included by appending a `1` to x_i
    and y_i before taking the outer product.
    :param x: Shape [batch_size, feature_x_dim]
    :param y: Shape [batch_size, feature_y_dim]
    :return: Shape [batch_size, (feature_x_dim + 1) * (feature_y_dim + 1)
    """
    batch_size, x_dim = x.size()
    _, y_dim = x.size()
    x = torch.cat((x, torch.ones(batch_size, 1)), dim=1)
    y = torch.cat((y, torch.ones(batch_size, 1)), dim=1)
    x_dim += 1
    y_dim += 1
    x = x.unsqueeze(2)
    y = y.unsqueeze(1)

    outer_prod = (
            x.expand(batch_size, x_dim, y_dim) * y.expand(batch_size, x_dim,
                                                          y_dim)
    )
    return outer_prod.view(batch_size, -1)


def batch_diag(diag_values, diag_mask=None):
    batch_size, dim = diag_values.size()
    if diag_mask is None:
        diag_mask = torch.diag(torch.ones(dim))
    batch_diag_mask = diag_mask.unsqueeze(0).expand(batch_size, dim, dim)
    batch_diag_values = diag_values.unsqueeze(1).expand(batch_size, dim, dim)
    return batch_diag_values * batch_diag_mask


def batch_square_vector(vector, M):
    """
    Compute x^T M x
    """
    vector = vector.unsqueeze(2)
    return torch.bmm(torch.bmm(vector.transpose(2, 1), M), vector).squeeze(2)


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def fanin_init_weights_like(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    new_tensor = FloatTensor(tensor.size())
    new_tensor.uniform_(-bound, bound)
    return new_tensor


def almost_identity_weights_like(tensor):
    """
    Set W = I + lambda * Gaussian no
    :param tensor:
    :return:
    """
    shape = tensor.size()
    init_value = np.eye(*shape)
    init_value += 0.01 * np.random.rand(*shape)
    return FloatTensor(init_value)


def clip1(x):
    return torch.clamp(x, -1, 1)


def compute_conv_output_size(h_in, w_in, kernel_size, stride, padding=0):
    h_out = (h_in + 2 * padding - (kernel_size - 1) - 1) / stride + 1
    w_out = (w_in + 2 * padding - (kernel_size - 1) - 1) / stride + 1
    return int(np.floor(h_out)), int(np.floor(w_out))


def compute_deconv_output_size(h_in, w_in, kernel_size, stride, padding=0):
    h_out = (h_in - 1) * stride - 2 * padding + kernel_size
    w_out = (w_in - 1) * stride - 2 * padding + kernel_size
    return int(np.floor(h_out)), int(np.floor(w_out))


def compute_conv_layer_sizes(h_in, w_in, kernel_sizes, strides, paddings=None):
    if paddings == None:
        for kernel, stride in zip(kernel_sizes, strides):
            h_in, w_in = compute_conv_output_size(h_in, w_in, kernel, stride)
            print('Output Size:', (h_in, w_in))
    else:
        for kernel, stride, padding in zip(kernel_sizes, strides, paddings):
            h_in, w_in = compute_conv_output_size(h_in, w_in, kernel, stride,
                                                  padding=padding)
            print('Output Size:', (h_in, w_in))


def compute_deconv_layer_sizes(h_in, w_in, kernel_sizes, strides,
                               paddings=None):
    if paddings == None:
        for kernel, stride in zip(kernel_sizes, strides):
            h_in, w_in = compute_deconv_output_size(h_in, w_in, kernel, stride)
            print('Output Size:', (h_in, w_in))
    else:
        for kernel, stride, padding in zip(kernel_sizes, strides, paddings):
            h_in, w_in = compute_deconv_output_size(h_in, w_in, kernel, stride,
                                                    padding=padding)
            print('Output Size:', (h_in, w_in))


"""
GPU wrappers
"""

_use_gpu = False
device = None


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")


def gpu_enabled():
    return _use_gpu


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


# noinspection PyPep8Naming
def FloatTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.FloatTensor(*args, **kwargs, device=torch_device)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def randint(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randint(*sizes, **kwargs, device=torch_device)


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)


def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)


def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)


def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)


def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)
