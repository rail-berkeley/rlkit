import re

import numpy as np
import torch


# "get_parameters" and "FreezeParameters" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
class FreezeParameters:
    def __init__(self, params):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because
        gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
                        output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.params = params
        self.param_states = [param.requires_grad for param in params]

    def __enter__(self):
        for param in self.params:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.params):
            param.requires_grad = self.param_states[i]


@torch.jit.script
def lambda_return(reward, value, discount, bootstrap, lambda_: float = 0.95):
    # from: https://github.com/yusukeurakami/dreamer-pytorch
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    """
    Compute the discounted reward for a batch of data.
    arguments:
        reward: [horizon - 1, batch, 1]
        value: [horizon - 1, batch, 1]
        discount: [horizon - 1, batch, 1]
        bootstrap: [batch, 1]
    returns:
        returns: [horizon - 1, batch, 1]
    """
    assert reward.shape[0] == value.shape[0] == discount.shape[0]
    assert reward.shape[1] == value.shape[1] == discount.shape[1]
    assert reward.shape[1] == bootstrap.shape[0]
    assert reward.shape[0] > 0
    next_values = torch.cat([value[1:], bootstrap.unsqueeze(0)], 0)
    target = reward + discount * next_values * (1 - lambda_)
    outputs = []
    accumulated_reward = bootstrap
    for t in range(reward.shape[0] - 1, -1, -1):
        inp = target[t]
        discount_factor = discount[t]
        accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
        outputs.append(accumulated_reward)
    returns = torch.flip(torch.stack(outputs), [0])
    return returns


@torch.jit.script
def compute_weights_from_discount(discount):
    weights = torch.cumprod(
        torch.cat(
            [
                torch.ones_like(discount[:1]),
                discount[:-1],
            ],
            0,
        ),
        0,
    )[:-1]
    return weights.detach()


def update_network(network, optimizer, gradient_clip, scaler):
    """
    Update the network parameters.
    Assume that loss.backward has already been called.
    """
    if isinstance(network, list):
        parameters = []
        for net in network:
            parameters.extend(list(net.parameters()))
    else:
        parameters = list(network.parameters())
    if gradient_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, norm_type=2)
    scaler.step(optimizer)
    optimizer.zero_grad(set_to_none=True)


# from dreamer_v2 repo
def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = np.clip(step / duration, 0, 1)
            return (1 - mix) * initial + mix * final
        match = re.match(r"warmup\((.+),(.+)\)", string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = np.clip(step / warmup, 0, 1)
            return scale * value
        match = re.match(r"exp\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        raise NotImplementedError(string)


def get_batch_length_indices(max_path_length, batch_length, batch_size):
    batch_start = np.random.randint(
        0, max_path_length - batch_length + 1, size=(batch_size)
    )
    batch_indices = np.linspace(
        batch_start,
        batch_start + batch_length,
        batch_length,
        endpoint=False,
    ).astype(int)
    return batch_indices


def get_indexed_arr_from_batch_indices(arr, batch_indices):
    """
    Faster alternative to:
    indexed_arr = np.take_along_axis(
                    arr,
                    np.expand_dims(batch_indices, -1),
                    axis=1,
                )
    2x speedup over take_along_axis.
    """
    batch_size = batch_indices.shape[1]
    if isinstance(arr, torch.Tensor):
        indexed_arr = arr[np.arange(batch_size), batch_indices].permute(1, 0, 2)
    else:
        indexed_arr = arr[np.arange(batch_size), batch_indices].transpose(1, 0, 2)
    return indexed_arr


def subsample_array_across_batch_length(
    arr, max_path_length, batch_length, batch_size, return_batch_indices=False
):
    """
    For each batch index, sample a subsequence of the second dimension.
    """
    batch_indices = get_batch_length_indices(max_path_length, batch_length, batch_size)
    indexed_arr = get_indexed_arr_from_batch_indices(arr, batch_indices)
    if return_batch_indices:
        return indexed_arr, batch_indices
    else:
        return indexed_arr


def save_env(env, path):
    import pickle

    pickle.dump(env, open(path, "wb"))


def load_env(path):
    import pickle

    env = pickle.load(open(path, "rb"))
    return env


def save_vec_env(path, env_suite, env_name, env_kwargs, n_envs, make_env):
    import pickle

    env_kwargs["n_envs"] = n_envs
    env_kwargs["make_env"] = make_env
    env_kwargs["env_name"] = env_name
    env_kwargs["env_suite"] = env_suite
    pickle.dump(env_kwargs, open(path, "wb"))


def load_vec_env(path):
    """
    Since we cannot pickle a vec env directly, we have to rebuild it from the env_kwargs.
    This will lose saved elements of the vec env / reset them.
    """
    import pickle

    from rlkit.envs.wrappers.mujoco_vec_wrappers import StableBaselinesVecEnv

    env_kwargs = pickle.load(open(path, "rb"))
    make_env = env_kwargs["make_env"]
    n_envs = env_kwargs["n_envs"]
    env_name = env_kwargs["env_name"]
    env_suite = env_kwargs["env_suite"]
    del env_kwargs["make_env"]
    del env_kwargs["n_envs"]
    del env_kwargs["env_name"]
    del env_kwargs["env_suite"]

    env_fns = [lambda: make_env(env_suite, env_name, env_kwargs) for _ in range(n_envs)]
    env = StableBaselinesVecEnv(env_fns=env_fns, start_method="fork")
    return env
