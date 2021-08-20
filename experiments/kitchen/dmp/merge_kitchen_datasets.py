
import h5py
import numpy as np
from tqdm import tqdm

import h5py
filename = "spirl/data/kitchen-vision/kitchen-total-v0-vision-64.hdf5"

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys
dataset = {}
with h5py.File(filename, 'r') as dataset_file:
    for k in tqdm(get_keys(dataset_file), desc="load datafile"):
        try:  # first try loading as an array
            dataset[k] = dataset_file[k][:]
        except ValueError as e:  # try loading as a scalar
            dataset[k] = dataset_file[k][()]
del dataset['images']
print(dataset.keys())
save_filename = '%s.hdf5' % 'kitchen-total-v0-state'
print('Saving dataset to %s.' % save_filename)
h5_dataset = h5py.File(save_filename, 'w')
for key in dataset:
    h5_dataset.create_dataset(key, data=dataset[key], compression='gzip')
print('Done.')
