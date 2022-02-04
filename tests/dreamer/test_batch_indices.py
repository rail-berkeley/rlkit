import numpy as np

from rlkit.torch.model_based.dreamer.utils import (
    get_batch_length_indices,
    subsample_array_across_batch_length,
)


def test_batch_indices_single_batch():
    max_path_length = 25
    batch_length = 25
    batch_size = 1
    batch_indices = get_batch_length_indices(
        max_path_length, batch_length, batch_size
    ).transpose(1, 0)
    assert batch_indices.shape == (batch_size, batch_length)
    assert np.all(batch_indices[0, :] == np.arange(batch_length))


def test_batch_indices_multi_batch():
    max_path_length = 25
    batch_length = 25
    batch_size = 10
    batch_indices = get_batch_length_indices(
        max_path_length, batch_length, batch_size
    ).transpose(1, 0)
    assert batch_indices.shape == (batch_size, batch_length)
    for i in range(batch_size):
        assert np.all(batch_indices[i, :] == np.arange(batch_length))


def test_subsample_array_across_batch_length():
    arr = np.random.rand(*(50, 500, 64 * 64 * 3))
    max_path_length = 500
    batch_length = 50
    batch_size = 50
    indexed_arr, batch_indices = subsample_array_across_batch_length(
        arr, max_path_length, batch_length, batch_size, return_batch_indices=True
    )
    tests = True
    for i in range(batch_size):
        test = (arr[i][batch_indices[:, i]] == indexed_arr[i]).all()
        tests = tests and test
    assert tests
