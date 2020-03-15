import numpy as np
import h5py
from tqdm import tqdm
import tensorflow as tf
from src.util_np import sample

def batch(path, path_cond, batch_size, seed=26, channel_first=False):
    """batch function to use with pipe"""
    ds = h5py.File(path, 'r')
    cond = np.load(path_cond, allow_pickle=True)["colors"]
    data = ds["data"]
    b, bc = [], []
    for i in sample(len(data), seed):
        if batch_size == len(b):
            yield np.array(b, dtype=np.float32), bc
            b, bc = [], []
        if channel_first: b.append(data[i].astype(np.float32)/255)
        else: b.append(np.rollaxis(data[i], 0, 3).astype(np.float32)/255)
        bc.append(cond[i])

def pipe(generator, output_types, prefetch=1, repeat=-1, name='pipe', **kwargs):
    """see `tf.data.Dataset.from_generator`."""
    return tf.data.Dataset.from_generator(generator, output_types) \
                          .repeat(repeat) \
                          .prefetch(prefetch) \
                          .__iter__()
