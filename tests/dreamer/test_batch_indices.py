import numpy as np

from rlkit.torch.model_based.dreamer.utils import get_batch_indices


def test_batch_indices_single_batch():
    max_path_length = 25
    batch_length = 25
    batch_size = 1
    batch_indices = get_batch_indices(max_path_length, batch_length, batch_size)
    assert batch_indices.shape == (batch_size, batch_length)
    assert np.all(batch_indices[0, :] == np.arange(batch_length))


def test_batch_indices_multi_batch():
    max_path_length = 25
    batch_length = 25
    batch_size = 10
    batch_indices = get_batch_indices(max_path_length, batch_length, batch_size)
    assert batch_indices.shape == (batch_size, batch_length)
    for i in range(batch_size):
        assert np.all(batch_indices[i, :] == np.arange(batch_length))
