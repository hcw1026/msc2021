# MIT License

# Copyright (c) 2020 Jin Xu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ============================================================================

# This file includes code from the project github.com/deepmind/leo.

########        Documented Changes        ########

# 1. This file mainly contains functions from github.com/deepmind/leo/data.py
# 2. Function construct_examples_batch is added here.
# 3. Command line flags are added here.

########        Original License for LEO        ########

# Copyright 2018 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from functools import partial
import numpy as np
import tensorflow as tf
import os
import pickle
import collections
from data.tools import ClassificationDescription, unpack_data, StrEnum, MetaSplit, GenerateDataset, normalise, description_map, shuffle_map

NDIM = 640

class MetaDataset(StrEnum):
      """datasets supported by the DataProvider class"""
      MINI = "miniImageNet"
      TIERED = "tieredImageNet"


class EmbeddingCrop(StrEnum):
    """embedding types supported by the DataProvider class"""
    CENTER = "center"
    MULTIVIEW = "multiview"
    

class DataProvider():
    """creates problem instances from a specific split and dataset"""
    def __init__(self, dataset_split, config):
        self._dataset_split = MetaSplit(dataset_split)
        if self._dataset_split == MetaSplit.TRAIN:
            self._batch_size = config["Train"]["batch_size"]
        else:
            self._batch_size = config["Eval"]["batch_size"]

        # ImageNet-specific Configurations
        _config = config["Data"]["leo_imagenet"]
        self._num_classes = _config["num_classes"]
        self._data_path = _config["data_path"]
        self._embedding_crop = _config["embedding_crop"]
        self._tr_size = _config["num_tr_examples_per_class"]
        self._val_size = _config["num_val_examples_per_class"]
        self._model_cls = _config["model_cls"]
        self._dataset_name = _config["dataset_name"]
        self._shuffle = _config["shuffle"]
        assert self._dataset_name.lower() in ['miniimagenet', 'tieredimagenet'], "Unknown dataset name"
        self._dataset_name = 'miniImageNet' if self._dataset_name.lower() == 'miniimagenet' else 'tieredImageNet'

        self._train_on_val = config["Train"]["train_on_val"]

        self._float_dtype = tf.float32
        self._int_dtype = tf.int32

        # Update and check config
        self._dataset_name = MetaDataset(self._dataset_name)
        self._embedding_crop = EmbeddingCrop(self._embedding_crop)
        if self._dataset_name == MetaDataset.TIERED:
            error_message = "embedding_crop: {} not supported for {}".format(
                self._embedding_crop, self._dataset_name)
            assert self._embedding_crop == EmbeddingCrop.CENTER, error_message

        # Initialise
        self._all_class_images = collections.OrderedDict() # dictionary with keys labels and values filenames
        self._image_embedding = collections.OrderedDict() # dictionary with keys filenames and values embeddings

        # Load data
        self._index_data(self._load_data())

    def _load(self, opened_file):
        """load pickle"""
        return pickle.load(opened_file, encoding="latin1")

    def _get_full_pickle_path(self, split_name):
        """get pickle path"""
        full_pickle_path = os.path.join(
            self._data_path,
            str(self._dataset_name),
            str(self._embedding_crop),
            "{}_embeddings.pkl".format(split_name))
        return full_pickle_path

    def _load_data(self):
        """loads data into memory and caches"""
        raw_data = self._load(tf.io.gfile.GFile(self._get_full_pickle_path(self._dataset_split), "rb"))

        if self._dataset_split == MetaSplit.TRAIN and self._train_on_val:
            valid_data = self._load(tf.io.gfile.GFile(self._get_full_pickle_path(MetaSplit.VALID), "rb"))

            # Merge train and validation dataset
            for key in valid_data:
                raw_data[key] = np.concatenate([raw_data[key],valid_data[key]], axis=0)

        return raw_data

    def _index_data(self, raw_data):
        """builds an index of images embeddings by class"""

        for idx, k in enumerate(raw_data["keys"]):
            if isinstance(k, bytes):
                k = k.decode("utf-8")

            # Extract class labels
            _, class_label, image_file = k.split("-")
            image_file_class_label = image_file.split("_")[0]
            assert class_label == image_file_class_label

            self._image_embedding[image_file] = raw_data["embeddings"][idx]
            if class_label not in self._all_class_images:
                self._all_class_images[class_label] = []
            self._all_class_images[class_label].append(image_file)

        # Check downloaded data is correct
        self._check_data_index(raw_data)

        self._all_class_images = collections.OrderedDict([
            (k, np.array(v)) for k, v in self._all_class_images.items()])

    def _check_data_index(self, raw_data):
        """performs checks of the data index and image counts per class"""
        n = raw_data["keys"].shape[0]
        error_message = "{} != {}".format(len(self._image_embedding), n)
        assert len(self._image_embedding) == n, error_message
        error_message = "{} != {}".format(raw_data["embeddings"].shape[0], n)
        assert raw_data["embeddings"].shape[0] == n, error_message

        all_class_folders = list(self._all_class_images.keys())
        error_message = "no duplicate class names"
        assert len(set(all_class_folders)) == len(all_class_folders), error_message
        image_counts = set([len(class_images)
                            for class_images in self._all_class_images.values()])
        error_message = ("len(image_counts) should have at least one element but "
                        "is: {}").format(image_counts)
        assert len(image_counts) >= 1, error_message
        assert min(image_counts) > 0


    def _generator(self):
        """generator for a random N-way K-shot classification problem instance"""
        def generator():
            """samples a random N-way K-shot classification problem instance"""
            # Randomly select num_classes classes
            class_list = list(self._all_class_images.keys()).copy()
            sample_count = (self._tr_size + self._val_size) # number of sample required for each class
            np.random.shuffle(class_list)
            shuffled_folders = class_list[:self._num_classes]
            
            error_message = "len(shuffled_folders) {} is not num_classes: {}".format(
                len(shuffled_folders), self._num_classes)
            assert len(shuffled_folders) == self._num_classes, error_message

            # Randomly select sample_count instances for each class
            image_paths = []
            class_ids = []
            embeddings = self._image_embedding
            for class_id, class_name in enumerate(shuffled_folders):
                all_images = self._all_class_images[class_name] # get filenames
                all_images = np.random.choice(all_images, sample_count, replace=False)

                error_message = "{} == {} failed".format(len(all_images), sample_count)
                assert len(all_images) == sample_count, error_message

                image_paths.append(all_images)
                class_ids.append([[class_id]]*sample_count)

            label_array = np.array(class_ids, dtype=np.int32) # with shape num_classes*sample_count*1
            path_array = np.array(image_paths) # with shape num_classes*sample_count
            embedding_array = np.array([[embeddings[image_path] for image_path in class_paths] for class_paths in path_array]) #with shape num_classes*sample_count*NDIM

            # Normalise and split into inner training and validations datasets
            #embedding_array /= np.linalg.norm(embedding_array, axis=-1,keepdims=True)

            tr_input, val_input = np.split(embedding_array, [self._tr_size], axis=1)
            tr_output, val_output = np.split(label_array, [self._tr_size], axis=1)
            tr_info, val_info = np.split(path_array, [self._tr_size], axis=1)

            tr_input = tf.reshape(tf.constant(tr_input, dtype=self._float_dtype), [self._tr_size*self._num_classes, -1])
            tr_output = tf.reshape(tf.constant(tr_output, dtype=self._int_dtype), [self._tr_size*self._num_classes, -1])
            tr_info = tf.reshape(tf.constant(tr_info, dtype=tf.string), [self._tr_size*self._num_classes])
            val_input = tf.reshape(tf.constant(val_input, dtype=self._float_dtype), [self._val_size*self._num_classes, -1])
            val_output = tf.reshape(tf.constant(val_output, dtype=self._int_dtype), [self._val_size*self._num_classes, -1])
            val_info = tf.reshape(tf.constant(val_info, dtype=tf.string), [self._val_size*self._num_classes])

            yield tr_input, tr_output, val_input, val_output

        return generator


    def generate(self):
        """generate dataset"""
        output_signature = (
        tf.TensorSpec(shape=(self._tr_size*self._num_classes, NDIM), dtype=self._float_dtype),
        tf.TensorSpec(shape=(self._tr_size*self._num_classes, 1), dtype=self._int_dtype),
        tf.TensorSpec(shape=(self._val_size*self._num_classes, NDIM), dtype=self._float_dtype),
        tf.TensorSpec(shape=(self._val_size*self._num_classes, 1), dtype=self._int_dtype)
        )

        dataset = tf.data.Dataset.from_generator(self._generator(), output_signature=output_signature)
        dataset = dataset.repeat()
        if self._shuffle is True:
            dataset = dataset.map(shuffle_map, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self._batch_size, drop_remainder=False) # batch
        dataset = dataset.map(normalise, num_parallel_calls=tf.data.AUTOTUNE) #normalise
        dataset = dataset.map(partial(description_map, description=ClassificationDescription), 
            num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils import parse_config
    config = parse_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/debug.yaml"))
    dataloader = DataProvider("trial", config)
    data = dataloader.generate()
    for i in data.take(1):
        pass