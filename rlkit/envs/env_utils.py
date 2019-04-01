import os

from gym.spaces import Box, Discrete, Tuple

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


def get_asset_full_path(file_name):
    return os.path.join(ENV_ASSET_DIR, file_name)


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))


def mode(env, mode_type):
    try:
        getattr(env, mode_type)()
    except AttributeError:
        pass
