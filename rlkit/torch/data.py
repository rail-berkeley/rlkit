import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# TODO: move this to more reasonable place
from rlkit.data_management.obs_dict_replay_buffer import normalize_image


class ImageDataset(Dataset):

    def __init__(self, images, should_normalize=True):
        super().__init__()
        self.dataset = images
        self.dataset_len = len(self.dataset)
        assert should_normalize == (images.dtype == np.uint8)
        self.should_normalize = should_normalize

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idxs):
        samples = self.dataset[idxs, :]
        if self.should_normalize:
            samples = normalize_image(samples)
        return np.float32(samples)


class InfiniteRandomSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.iter = iter(torch.randperm(len(self.data_source)).tolist())

    def __iter__(self):
        return self

    def __next__(self):
        try:
            idx = next(self.iter)
        except StopIteration:
            self.iter = iter(torch.randperm(len(self.data_source)).tolist())
            idx = next(self.iter)
        return idx

    def __len__(self):
        return 2 ** 62


class InfiniteWeightedRandomSampler(Sampler):

    def __init__(self, data_source, weights):
        assert len(data_source) == len(weights)
        assert len(weights.shape) == 1
        self.data_source = data_source
        # Always use CPU
        self._weights = torch.from_numpy(weights)
        self.iter = self._create_iterator()

    def update_weights(self, weights):
        self._weights = weights
        self.iter = self._create_iterator()

    def _create_iterator(self):
        return iter(
            torch.multinomial(
                self._weights, len(self._weights), replacement=True
            ).tolist()
        )

    def __iter__(self):
        return self

    def __next__(self):
        try:
            idx = next(self.iter)
        except StopIteration:
            self.iter = self._create_iterator()
            idx = next(self.iter)
        return idx

    def __len__(self):
        return 2 ** 62
