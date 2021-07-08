import copy
import tensorflow as tf
import collections
import enum
from functools import partial

import os
import h5py
import numpy as np


###########################################################################################
# All
###########################################################################################
class StrEnum(enum.Enum):
    """an Enum represented by a string"""

    def __str__(self):
          return self.value

    def __repr__(self):
         return self.__str__()

class MetaSplit(StrEnum):
    """meta-datasets split supported by the DataProvider class"""
    TRAIN = "train"
    VALID = "val"
    TEST = "test"
    TRIAL = "trial" # for debugging


###########################################################################################
# Classification
###########################################################################################
ClassificationDescription = collections.namedtuple(
    "ClassificationDescription",
    ["tr_input", "tr_output", "val_input", "val_output"])

def unpack_data(problem_instance):
    if isinstance(problem_instance, ClassificationDescription):
        return list(problem_instance)
    return problem_instance

def normalise(tr_input, tr_output, val_input, val_output):
    """normalise tr.input and val.input independently"""
    tr_input = tf.nn.l2_normalize(tr_input, axis=-1)
    val_input = tf.nn.l2_normalize(val_input, axis=-1)
    return tr_input, tr_output, val_input, val_output

def description_map(tr_input, tr_output, val_input, val_output, description):
    """map list into description"""
    return description(tr_input, tr_output, val_input, val_output)

def shuffle_map(tr_input, tr_output, val_input, val_output):
    """shuffle within problem instance"""
    tr_size = tf.shape(tr_input)[0]
    val_size = tf.shape(val_input)[0]
    
    tr_indices = tf.random.shuffle(tf.range(0, tr_size, dtype=tf.int32))
    val_indices = tf.random.shuffle(tf.range(0, val_size, dtype=tf.int32))

    tr_input = tf.gather(tr_input, tr_indices)
    tr_output = tf.gather(tr_output, tr_indices)
    val_input = tf.gather(val_input, val_indices)
    val_output = tf.gather(val_output, val_indices)
    return tr_input, tr_output, val_input, val_output

def GenerateDataset(generator, output_signature, batch_size, description, shuffle):
    """data input pipeline for a generated tf.data.Dataset"""
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.repeat()
    if shuffle is True:
        dataset = dataset.map(shuffle_map, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False) # batch
    dataset = dataset.map(normalise, num_parallel_calls=tf.data.AUTOTUNE) #normalise
    dataset = dataset.map(partial(description_map, description=description), 
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


###########################################################################################
# Regression
###########################################################################################
RegressionDescription = collections.namedtuple(
    "RegressionDescription",
    ["tr_input", "tr_output", "val_input", "val_output"])

#### Datasets utilities
class DatasetMerger():
    """
    Helper which merges an iterable of datasets. Assume that they all have the same attributes 
    (redirect to the first one).
    """

    def __init__(self, datasets, shuffle=True):
        datasets = copy.deepcopy(datasets)
        self.datasets_names, self.datasets = list(zip(*datasets.items()))
        self.cumul_len = np.cumsum([len(d) for d in self.datasets])
        self.shuffle_map = np.random.permutation(np.arange(self.cumul_len[-1]))
        self.shuffle = shuffle

    def __getitem__(self, index):
        if not self.__getattr__("is_reuse_across_epochs"):
            index = index // self.cumul_len[-1] # allow generator dataset
        
        if self.shuffle:
            index = self.shuffle_map[index]

        idx_dataset = self.cumul_len.searchsorted(index + 1)  # + 1 because of 0 index
        idx_in_dataset = index
        if idx_dataset > 0:
            idx_in_dataset -= self.cumul_len[idx_dataset - 1]  # - 1 because rm previous
        return self.datasets[idx_dataset][idx_in_dataset]

    def __len__(self):
        return self.cumul_len[-1]

    def __getattr__(self, attr):
        return getattr(self.datasets[0], attr)

#### other utilities
def rescale_range(X, old_range, new_range):
    """Rescale X linearly to be in `new_range` rather than `old_range`."""
    old_min = old_range[0]
    new_min = new_range[0]
    old_delta = old_range[1] - old_min
    new_delta = new_range[1] - new_min
    return (((X - old_min) * new_delta) / old_delta) + new_min

class NotLoadedError(Exception):
    pass

#### save files
def _parse_save_file_chunk(save_file, idx_chunk):
    if save_file is None:
        save_file, save_group = None, None
    elif isinstance(save_file, tuple):
        save_file, save_group = save_file[0], save_file[1] + "/"
    elif isinstance(save_file, str):
        save_file, save_group = save_file, ""
    else:
        raise ValueError("Unsupported type of save_file={}.".format(save_file))

    if idx_chunk is not None:
        chunk_suffix = "_chunk_{}".format(idx_chunk)
    else:
        chunk_suffix = ""

    return save_file, save_group, chunk_suffix

def load_chunk(keys, save_file, idx_chunk):
    items = dict()
    save_file, save_group, chunk_suffix = _parse_save_file_chunk(save_file, idx_chunk)

    if save_file is None or not os.path.exists(save_file):
        raise NotLoadedError()

    try:
        with h5py.File(save_file, "r") as hf:
            for k in keys:
                items[k] = tf.constant(hf["{}{}{}".format(save_group, k, chunk_suffix)][:])
    except KeyError:
        raise NotLoadedError()

    return items

def save_chunk(to_save, save_file, idx_chunk):
    save_file, save_group, chunk_suffix = _parse_save_file_chunk(save_file, idx_chunk)

    if save_file is None:
        return  # don't save

    print("Saving group {} chunk {} for future use ...".format(save_group, idx_chunk))

    with h5py.File(save_file, "a") as hf:
        for k, v in to_save.items():
            hf.create_dataset(
                "{}{}{}".format(save_group, k, chunk_suffix), data=v.numpy()
            )

def ratio_to_int(percentage, max_val):
    """Converts a ratio to an integer if it is smaller than 1."""
    if 1 <= percentage <= max_val:
        out = percentage
    elif 0 <= percentage < 1:
        out = percentage * max_val
    else:
        raise ValueError("percentage={} outside of [0,{}].".format(percentage, max_val))

    return int(out)

def indep_shuffle_(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply `numpy.random.shuffle` to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.

    Credits : https://github.com/numpy/numpy/issues/5173
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])

def convert_indices(indices):
    """convert context getter indices into tf.gather_nd compatible indices"""
    batch_size, num_points = indices.shape
    dim_indices =  tf.tile(tf.expand_dims(tf.range(batch_size),-1),(1,num_points))
    return tf.stack([dim_indices, indices],axis=-1)