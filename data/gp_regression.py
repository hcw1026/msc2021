# MIT License

# Copyright (c) 2020 Yann Dubois, Jonathan Gordon, Andrew YK Foong

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

import copy
import numpy as np
import os
import random
from scipy.stats import betabinom
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, Matern

from data.tools import *

import tensorflow as tf

DIR_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Other/gp/"))

###########################################################################################
# DataProvider
###########################################################################################
class DataProvider():
    def __init__(self, config, load_type="all", custom_kernels=None, custom_kernels_merge=False, train_datasets=None, train_batch_size=None, eval_batch_size=None, train_indp_target=None, eval_indp_target=None, **kwargs):
        """data loader for GP 1D regression training
        config: dict
            - configuration file - the same format as a parsed yaml in the config/sample.yaml
        load_type: str, one of "all", "single", "var_hyp", "var_kernel", "custom"
            - single contains three kernels with fixed parameters: RBF, Periodic, noisy Matern (see get_datasets_single_gp)
              var_hyp contains Matern with variable lengthscale (see get_datasets_variable_hyp_gp)
              var_kernel contains a mixture of three kernels with fixed parameters: RBF, Periodic, noisy Matern (see get_datasets_variable_kernel_gp)
        custom_kernels: dict of sklearn.gaussian_process.kernels
            - if load_type is custom, the dataset is generated using the custom_kernels supplied. Use **kwargs to overwrite the config. See GPDatasets and sklearn.GaussianProcessProcessRegressor for what parameters available
        custom_kernels_merge: bool
            - if True, DatasetMerger is used to merge the computed datasets using custom_kernels
        train_batch_size: int or None
            - the batch size of training datasets (from .generate() method). If None, the value is taken from config
        eval_batch_size int or None
            - the batch size of validation and test datasets (from .generate() and .generate_test() method). If None, the value is taken from config
        train_indp_target: bool, optional
            - If True, any overlapping context points are removed in the set of target points for training set. If None, the value is taken from config
        eval_indp_target: bool, optional
            - If True, any overlapping context points are removed in the set of target points for validation and test datasets. If None, the value is taken from config
        """
        self._train_datasets = train_datasets
        self._load_type = load_type
        self._custom_kernels = custom_kernels
        self._custom_kernels_merge = custom_kernels_merge

        # Training and eval Configurations
        self._train_drop_remainder = config["Train"]["drop_remainder"]
        self._eval_drop_remainder = config["Eval"]["drop_remainder"]
        self._train_batch_size = train_batch_size if train_batch_size is not None else config["Train"]["batch_size"]
        self._eval_batch_size = eval_batch_size if eval_batch_size is not None else config["Eval"]["batch_size"]

        # Dataset Configurations
        _config = config["Data"]["gp_regression"]
        self._shuffle = read_kwargs(kwargs=kwargs, config=_config, config_name="shuffle", kwargs_name="shuffle", delete=True)
        self._save_path = read_kwargs(kwargs=kwargs, config=_config, config_name="save_path", kwargs_name="save_file", delete=True)
        
        self._n_points = read_kwargs(kwargs=kwargs, config=_config, config_name="n_points", kwargs_name="n_points", delete=True)
        self._n_samples = read_kwargs(kwargs=kwargs, config=_config, config_name="n_samples", kwargs_name="n_samples", delete=True)
        self._n_same_samples = read_kwargs(kwargs=kwargs, config=_config, config_name="n_same_samples", kwargs_name="n_same_samples", delete=True)
        self._is_reuse_across_epochs = read_kwargs(kwargs=kwargs, config=_config, config_name="is_reuse_across_epochs", kwargs_name="is_reuse_across_epochs", delete=True)

        self._val_n_samples = read_kwargs(kwargs=kwargs, config=_config, config_name="val_n_samples", kwargs_name="val_n_samples", delete=True)
        self._test_n_samples = read_kwargs(kwargs=kwargs, config=_config, config_name="test_n_samples", kwargs_name="test_n_samples", delete=True)

        self._min_context = read_kwargs(kwargs=kwargs, config=_config, config_name="min_context", kwargs_name="min_context", delete=True)
        self._max_context = read_kwargs(kwargs=kwargs, config=_config, config_name="max_context", kwargs_name="max_context", delete=True)
        self._is_batch_share = read_kwargs(kwargs=kwargs, config=_config, config_name="is_batch_share", kwargs_name="is_batch_share", delete=True)
        self._range_indcs = read_kwargs(kwargs=kwargs, config=_config, config_name="range_indcs", kwargs_name="range_indcs", delete=True)
        self._is_beta_binomial = read_kwargs(kwargs=kwargs, config=_config, config_name="is_beta_binomial", kwargs_name="is_beta_binomial", delete=True)
        self._proba_uniform = read_kwargs(kwargs=kwargs, config=_config, config_name="proba_uniform", kwargs_name="proba_uniform", delete=True)

        self._train_indp_target = train_indp_target if train_indp_target is not None else _config["train_indp_target"]
        self._eval_indp_target = eval_indp_target if eval_indp_target is not None else _config["eval_indp_target"]

        kwargs.update(dict(n_same_samples=self._n_same_samples))
        self.kwargs = kwargs

        # Other initialisation
        contexts_getter = GetRandomIndcs(
            a=self._min_context, 
            b=self._max_context, 
            is_batch_share=self._is_batch_share, 
            range_indcs=self._range_indcs,
            is_beta_binomial=self._is_beta_binomial, 
            proba_uniform=self._proba_uniform)
        self.splitter = CntxtTrgtGetter(contexts_getter=contexts_getter)
        self._float_dtype = tf.float32

        # Load data
        self.datasets = self._load(dataset_split="train", train_datasets=self._train_datasets, n_samples=self._n_samples, n_points=self._n_points, is_reuse_across_epochs=self._is_reuse_across_epochs)

    @property
    def get_datasets(self):
        """return GPDataset object"""
        return self.datasets

    def _load(self, dataset_split="train", train_datasets=None, n_samples=50000, n_points=128, is_reuse_across_epochs=False):
        if self._load_type.lower() == "all":
            return get_all_gp_datasets(dataset_split=dataset_split, train_datasets=train_datasets, n_samples=n_samples, n_points=n_points, is_reuse_across_epochs=is_reuse_across_epochs, save_file=self._save_path, **self.kwargs)
        elif self._load_type.lower() == "single":
            return get_datasets_single_gp(dataset_split=dataset_split, train_datasets=train_datasets, n_samples=n_samples, n_points=n_points, is_reuse_across_epochs=is_reuse_across_epochs, save_file=self._save_path, **self.kwargs)
        elif self._load_type.lower() == "var_hyp":
            return get_datasets_variable_hyp_gp(dataset_split=dataset_split, train_datasets=train_datasets, n_samples=n_samples, n_points=n_points, is_reuse_across_epochs=is_reuse_across_epochs, save_file=self._save_path, **self.kwargs)
        elif self._load_type.lower() == "var_kernel":
            return get_datasets_variable_kernel_gp(dataset_split=dataset_split, train_datasets=train_datasets, n_samples=n_samples, n_points=n_points, is_reuse_across_epochs=is_reuse_across_epochs, save_file=self._save_path, **self.kwargs)
        elif self._load_type.lower() == "custom":
            if self._custom_kernels is not None:
                if self._custom_kernels_merge:
                    return get_datasets_variable_kernel_gp(dataset_split=dataset_split, train_datasets=train_datasets, kernels=self._custom_kernels, n_samples=n_samples, n_points=n_points, is_reuse_across_epochs=is_reuse_across_epochs, save_file=self._save_path, **self.kwargs)
                else:
                    return get_datasets_single_gp(dataset_split=dataset_split, train_datasets=train_datasets, kernels=self._custom_kernels, n_samples=n_samples, n_points=n_points, is_reuse_across_epochs=is_reuse_across_epochs, save_file=self._save_path, **self.kwargs)
            else:
                raise ValueError("custom_kernels must be provided when load_type == 'custom'")
        else:
            raise NameError("_load_type is invalid")

    def _generator(self, dataset, batch_size, indp_target):
        """return self.datasets as generator"""
        def generator():
            # Batching
            X, y = list(zip(*[dataset[0] for i in range(batch_size)])) #simply take the first element as it is randomly generated anyway
            X = tf.stack(X)
            y = tf.stack(y)
            yield self.splitter(X=X, y=y, indp_target=indp_target)

        return generator

    def _generate_from_generator(self, dataset, batch_size, indp_target):

        output_signature = (
        tf.TensorSpec(shape=(batch_size, None, 1), dtype=self._float_dtype), 
        tf.TensorSpec(shape=(batch_size, None, 1), dtype=self._float_dtype),
        tf.TensorSpec(shape=(batch_size, None, 1), dtype=self._float_dtype),
        tf.TensorSpec(shape=(batch_size, None, 1), dtype=self._float_dtype))

        dataset_out = tf.data.Dataset.from_generator(
            generator=self._generator(dataset, batch_size=batch_size, indp_target=indp_target),
            output_signature=output_signature 
            )

        dataset_out = dataset_out.repeat()
        dataset_out = self._dataset_pipeline(dataset_out, batch_size)

        return dataset_out

    @tf.autograph.experimental.do_not_convert
    def _generate_from_dataset(self, dataset, batch_size, indp_target, drop_remainder):

        if isinstance(dataset, DatasetMerger):
            X = tf.concat([d[:][0] for d in dataset.datasets], axis=0)
            y = tf.concat([d[:][1] for d in dataset.datasets], axis=0)
        else:
            X, y = dataset[:]

        # Batching
        num_points = len(dataset)
        num_split, remainder = divmod(num_points,batch_size)
        X_remain = X[num_points-remainder:]
        X = tf.split(X[:num_points-remainder], num_or_size_splits=num_split, axis=0)
        y_remain = y[num_points-remainder:]
        y = tf.split(y[:num_points-remainder], num_or_size_splits=num_split, axis=0)
        if (not drop_remainder) and (X_remain.shape[0] > 0):
            X.append(X_remain)
            y.append(y_remain)

        # Context and target splitter
        tr_input, tr_output, val_input, val_output = (zip(*[self.splitter(X=X_, y=y_, indp_target=indp_target) for X_, y_ in zip(*[X,y])]))

        def map_fn(idx):
            return (tr_input[idx], tr_output[idx], val_input[idx], val_output[idx])

        dataset_out = tf.data.Dataset.range(len(X))
        dataset_out = dataset_out.map(lambda idx: tf.py_function(
            func=map_fn, 
            inp=[idx], 
            Tout=(tf.float32, tf.float32, tf.float32, tf.float32)))

        dataset_out = self._dataset_pipeline(dataset_out, batch_size)
  
        return dataset_out

    def _dataset_pipeline(self, dataset, batch_size):
        if self._shuffle:
            dataset = dataset.map(shuffle_map, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(partial(description_map, description=RegressionDescription), 
            num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.unbatch().batch(batch_size) #form tf.data batch
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def generate(self, datasets=None, return_valid=False, return_test=True, train_batch_size=None, eval_batch_size=None, train_indp_target=None, eval_indp_target=None, train_drop_remainder=None, eval_drop_remainder=None):
        """
        train_batch_size: int or None
            - the batch size of training datasets. If None, the value is taken from the class initialisation
        eval_batch_size int or None
            - the batch size of validatiSon and test datasets). If None, the value is taken from the class initialisation
        train_indp_target: bool, optional
            - If True, any overlapping context points are removed in the set of target points for training dataset. If None, the value is taken from the class initialisation
        eval_indp_target: bool, optional
            - If True, any overlapping context points are removed in the set of target points for validation and test datasets. If None, the value is taken from the class initialisation
        train_drop_remainder: bool, optional
            - If True, remainder from batches are dropped for training dataset. If None, the value is taken from the class initialisation
        eval_drop_remainder: bool, optional
            - If True, remainder from batches are dropped for validation and test dataset. If None, the value is taken from the class initialisation
        return in the order (training_data, validation_data, test_data) where appropriate
        """

        train_batch_size = train_batch_size if train_batch_size is not None else self._train_batch_size
        eval_batch_size = eval_batch_size if eval_batch_size is not None else self._eval_batch_size

        train_indp_target = train_indp_target if train_indp_target is not None else self._train_indp_target
        eval_indp_target = eval_indp_target if eval_indp_target is not None else self._eval_indp_target

        train_drop_remainder = train_drop_remainder if train_drop_remainder is not None else self._train_drop_remainder
        eval_drop_remainder = eval_drop_remainder if eval_drop_remainder is not None else self._eval_drop_remainder

        if datasets is None: # default generate from self.datasets
            datasets = self.datasets

        datasets_out = dict()

        for k, dataset in datasets.items():
            if dataset.is_reuse_across_epochs:
                datasets_out.update({k:self._generate_from_dataset(dataset, batch_size=train_batch_size, indp_target=train_indp_target, drop_remainder=train_drop_remainder)})
            else:
                datasets_out.update({k:self._generate_from_generator(dataset, batch_size=train_batch_size, indp_target=train_indp_target)})

        if not (return_valid or return_test):
            return datasets_out
        elif not return_valid:
            datasets_test = self.generate_test("test", train_datasets=datasets, n_samples=self._test_n_samples, batch_size=eval_batch_size, indp_target=eval_indp_target)
            return (datasets_out, datasets_test)
        elif not return_test:
            datasets_valid = self.generate_test("valid", train_datasets=datasets, n_samples=self._val_n_samples, batch_size=eval_batch_size, indp_target=eval_indp_target)
            return (datasets_out, datasets_valid)
        else:
            datasets_valid = self.generate_test("valid", train_datasets=datasets, n_samples=self._val_n_samples, batch_size=eval_batch_size, indp_target=eval_indp_target)
            datasets_test = self.generate_test("test", train_datasets=datasets, n_samples=self._test_n_samples, batch_size=eval_batch_size, indp_target=eval_indp_target)
            return (datasets_out, datasets_valid, datasets_test)

    def generate_test(self, dataset_split="test", train_datasets=None, n_samples=10000, batch_size=None, indp_target=None, drop_remainder=None):
        """generate test set from a training dataset GPDataset class
        dataset_split: str
            - "val" or "test". Generate a fixed size dataset with the same parameters as the training dataset of this class instance
        train_datasets:
            - the training datasets (dict) used to generate the test datasets. If not, the training dataset generated in this class is used
        n_samples: int
            - number of samples to generate
        batch_size: int or None
            - the batch size of the generated datasets. If None, the value is taken from the class eval_batch_size initialisation
        indp_target: bool, optional
            - If True, any overlapping context points are removed in the set of target points for training set. If None, the value is taken from the class eval_indp_target initialisation
        drop_remainder: bool, optional
            - If True, remainder of batches are dropped. If None, the value is taken from the class eval_drop_remainder initialisation
        """

        eval_batch_size = batch_size if batch_size is not None else self._eval_batch_size
        eval_indp_target = indp_target if indp_target is not None else self._eval_indp_target
        eval_drop_remainder = drop_remainder if drop_remainder is not None else self._eval_drop_remainder

        if train_datasets is None:
            train_datasets = self.datasets

        test_datasets = self._load(dataset_split=dataset_split, train_datasets=train_datasets, n_samples=n_samples, n_points=None, is_reuse_across_epochs=True)
        return self.generate(test_datasets, return_valid=False, return_test=False, train_batch_size=eval_batch_size, eval_batch_size=None, train_indp_target=eval_indp_target, eval_indp_target=None, train_drop_remainder=eval_drop_remainder, eval_drop_remainder=None)


###########################################################################################
# Dataset class
###########################################################################################
class GPDataset():
    """
    Dataset of functions generated by a gaussian process.

    Parameters
    ----------
    kernel : sklearn.gaussian_process.kernels or list
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default.

    min_max : tuple of floats, optional
        Min and max point at which to evaluate the function (bounds).

    n_samples : int, optional
        Number of sampled functions contained in dataset.

    n_points : int, optional
        Number of points at which to evaluate f(x) for x in min_max.

    is_vary_kernel_hyp : bool, optional
        Whether to sample each example from a kernel with random hyperparameters,
        that are sampled uniformly in the kernel hyperparameters `*_bounds`.

    save_file : string or tuple of strings, optional
        Where to save and load the dataset. If tuple `(file, group)`, save in
        the hdf5 under the given group. If `None` regenerate samples indefinitely.
        Note that if the saved dataset has been completely used,
        it will generate a new sub-dataset for every epoch and save it for future
        use.

    n_same_samples : int, optional
        Number of samples with same kernel hyperparameters and X. This makes the
        sampling quicker.

    is_reuse_across_epochs : bool, optional
        Whether to reuse the same samples across epochs.  This makes the
        sampling quicker and storing less memory heavy if `save_file` is given.

    generated_from : str or None, optional
        A string to record which function is used to generate this dataset instance

    kwargs:
        Additional arguments to `GaussianProcessRegressor`.
    """

    def __init__(
        self,
        kernel=(
            WhiteKernel(noise_level=0.1, noise_level_bounds=(0.1, 0.5))
            + RBF(length_scale=0.4, length_scale_bounds=(0.1, 1.0))
        ),
        min_max=(-2, 2),
        n_samples=1000,
        n_points=128,
        is_vary_kernel_hyp=False,
        save_file=None,
        n_same_samples=20,
        is_reuse_across_epochs=True,
        generated_from=None,
        kernel_name=None,
        **kwargs,
    ):

        self.n_samples = n_samples
        self.n_points = n_points
        self.min_max = min_max
        self.is_vary_kernel_hyp = is_vary_kernel_hyp
        self.save_file = save_file
        self.n_same_samples = n_same_samples
        self.is_reuse_across_epochs = is_reuse_across_epochs

        self.generated_from = generated_from if generated_from is not None else "custom"
        self.kernel_name = kernel_name

        self._float_dtype = tf.float32

        self._idx_precompute = 0  # current index of precomputed data
        self._idx_chunk = 0  # current chunk (i.e. epoch)

        if not is_vary_kernel_hyp:
            # only fit hyperparam when predicting if using various hyperparam
            kwargs["optimizer"] = None

            # we also fix the bounds as these will not be needed
            for hyperparam in kernel.hyperparameters:
                kernel.set_params(**{f"{hyperparam.name}_bounds": "fixed"})

        self.generator = GaussianProcessRegressor(
            kernel=kernel, alpha=0.005, **kwargs  # numerical stability for preds
        )

        self.precompute_chunk_()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.is_reuse_across_epochs:
            return self.data[index], self.targets[index]

        else:
            # doesn't use index because randomly generated in any case => sample
            # in order which enables to know when epoch is finished and regenerate
            # new functions
            self._idx_precompute += 1
            if self._idx_precompute == self.n_samples:
                self.precompute_chunk_()
            return self.data[self._idx_precompute], self.targets[self._idx_precompute]

    def get_samples(
        self,
        n_samples=None,
        test_min_max=None,
        n_points=None,
        save_file=None,
        idx_chunk=None,
        rounding=5
    ):
        """Return a batch of samples

        Parameters
        ----------
        n_samples : int, optional
            Number of sampled function (i.e. batch size). Has to be dividable
            by n_diff_kernel_hyp or 1. If `None` uses `self.n_samples`.

        test_min_max : float, optional
            Testing range. If `None` uses training one.

        n_points : int, optional
            Number of points at which to evaluate f(x) for x in min_max. If None
            uses `self.n_points`.

        save_file : string, optional
            Where to save and load the dataset. The file will be save din
            the hdf5 under the given group according to the dataset properties. If `None` uses does not save.

        idx_chunk : int, optional
            Index of the current chunk. This is used when `save_file` is not None,
            and you want to save a single dataset through multiple calls to
            `get_samples`.

        rounding: int, optional
            Decimal rounding used for test_min_max in saving signture
        """
        test_min_max = test_min_max if test_min_max is not None else self.min_max
        n_points = n_points if n_points is not None else self.n_points
        n_samples = n_samples if n_samples is not None else self.n_samples

        save_signature = generate_save_signature(kernel_name=self.kernel_name, min_max=test_min_max, n_points=self.n_points, is_vary_kernel_hyp=self.is_vary_kernel_hyp, n_same_samples=self.n_same_samples, rounding=rounding)

        save_file = get_save_file(name=save_signature, save_file=save_file)

        try:
            loaded = load_chunk({"data", "targets"}, save_file, idx_chunk, n_samples)
            data, targets = loaded["data"], loaded["targets"]
        except NotLoadedError:
            X = self._sample_features(test_min_max, n_points, n_samples)
            X, targets = self._sample_targets(X, n_samples)
            data = self._postprocessing_features(X, n_samples)
            save_chunk(
                {"data": data, "targets": targets},
                save_file,
                idx_chunk,
            )

        return data, targets

    def set_samples_(self, data, targets):
        """Use the samples (output from `get_samples`) as the data."""
        self.is_reuse_across_epochs = True
        self.data = data
        self.targets = targets
        self.n_samples = self.data.shape[0]

    def precompute_chunk_(self):
        """Load or precompute and save a chunk (data for an epoch.)"""
        self._idx_precompute = 0
        self.data, self.targets = self.get_samples(
            save_file=self.save_file, idx_chunk=self._idx_chunk
        )
        self._idx_chunk += 1

    def _sample_features(self, min_max, n_points, n_samples):
        """Sample X with non uniform intervals. """
        X = np.random.uniform(min_max[1], min_max[0], size=(n_samples, n_points))
        # sort which is convenient for plotting
        X.sort(axis=-1)
        return X

    def _postprocessing_features(self, X, n_samples):
        """Convert the features to a tensor, rescale them to [-1,1] and expand."""
        X = tf.expand_dims(tf.constant(X, dtype=self._float_dtype),-1)
        X = rescale_range(X, self.min_max, (-1, 1))
        return X

    def _sample_targets(self, X, n_samples):
        targets = X.copy()
        n_samples, n_points = X.shape
        for i in range(0, n_samples, self.n_same_samples):
            if self.is_vary_kernel_hyp:
                self.sample_kernel_()

            for attempt in range(self.n_same_samples):
                # can have numerical issues => retry using a different X
                try:
                    # takes care of boundaries
                    n_same_samples = targets[i : i + self.n_same_samples, :].shape[0]
                    targets[i : i + self.n_same_samples, :] = self.generator.sample_y(
                        X[i + attempt, :, np.newaxis],
                        n_samples=n_same_samples,
                        random_state=None,
                    ).transpose(1, 0)
                    X[i : i + self.n_same_samples, :] = X[i + attempt, :]
                except np.linalg.LinAlgError:
                    continue  # try again
                else:
                    break  # success
            else:
                raise np.linalg.LinAlgError("SVD did not converge 10 times in a row.")

        # shuffle output to not have n_same_samples consecutive
        X, targets = sklearn.utils.shuffle(X, targets)
        targets = tf.constant(targets, dtype=self._float_dtype)
        targets = tf.reshape(targets, [n_samples, n_points, 1])
        return X, targets

    def sample_kernel_(self):
        """
        Modify inplace the kernel hyperparameters through uniform sampling in their
        respective bounds.
        """
        K = self.generator.kernel
        for hyperparam in K.hyperparameters:
            K.set_params(
                **{hyperparam.name: np.random.uniform(*hyperparam.bounds.squeeze())}
            )


###########################################################################################
# Dataset splitter
###########################################################################################
def get_all_indcs(batch_size, n_possible_points):
    """
    Return all possible indices.
    """
    #torch.arange(n_possible_points).expand(batch_size, n_possible_points)
    return tf.tile(tf.expand_dims(tf.range(n_possible_points),0),(batch_size,1))

class GetRandomIndcs:
    """
    Return random subset of indices.

    Parameters
    ----------
    a : float or int, optional
        Minimum number of indices. If smaller than 1, represents a percentage of
        points.

    b : float or int, optional
        Maximum number of indices. If smaller than 1, represents a percentage of
        points.

    is_batch_share : bool, optional
        Whether to use use the same indices for all elements in the batch.

    range_indcs : tuple, optional
        Range tuple (max, min) for the indices.
        
    is_beta_binomial : bool, optional
        Whether to use beta binomial distribution instead of uniform. In this case a and b become
        respectively alpha and beta in beta binomial distributions. For example to have a an 
        exponentially decaying pdf with a median around 5% use alpha 1 and beta 14.

    proba_uniform : float, optional
        Probability [0,1] of randomly sampling any number of indices regardless of a and b. Useful to 
        ensure that the support is all possible indices.
    """

    def __init__(
        self,
        a=0.1,
        b=0.5,
        is_batch_share=False,
        range_indcs=None,
        is_ensure_one=False,
        is_beta_binomial=False,
        proba_uniform=0,
    ):
        self.a = a
        self.b = b
        self.is_batch_share = is_batch_share
        self.range_indcs = range_indcs
        self.is_ensure_one = is_ensure_one
        self.is_beta_binomial = is_beta_binomial
        self.proba_uniform = proba_uniform

    def __call__(self, batch_size, n_possible_points):
        if self.range_indcs is not None:
            n_possible_points = self.range_indcs[1] - self.range_indcs[0]

        if np.random.uniform(size=1) < self.proba_uniform:
            # whether to sample from a uniform distribution instead of using a and b
            n_indcs = random.randint(0, n_possible_points)

        else:
            if self.is_beta_binomial:
                rv = betabinom(n_possible_points, self.a, self.b)
                n_indcs = rv.rvs()

            else:
                a = ratio_to_int(self.a, n_possible_points)
                b = ratio_to_int(self.b, n_possible_points)
                n_indcs = random.randint(a, b)

        if self.is_ensure_one and n_indcs < 1:
            n_indcs = 1

        if self.is_batch_share:
            # indcs = torch.randperm(n_possible_points)[:n_indcs]
            # indcs = indcs.unsqueeze(0).expand(batch_size, n_indcs)
            indcs = tf.random.shuffle(tf.range(n_possible_points))[:n_indcs]
            indcs = tf.tile(tf.expand_dims(indcs, 0),(batch_size,1))
        else:
            indcs = (
                np.arange(n_possible_points)
                .reshape(1, n_possible_points)
                .repeat(batch_size, axis=0)
            )
            indep_shuffle_(indcs, -1)
            #indcs = torch.from_numpy(indcs[:, :n_indcs])
            indcs = tf.constant(indcs[:,:n_indcs])

        if self.range_indcs is not None:
            # adding is teh same as shifting
            indcs += self.range_indcs[0]

        return indcs

class CntxtTrgtGetter:
    """
    Split a dataset into context and target points based on indices.

    Parameters
    ----------
    contexts_getter : callable, optional
        Get the context indices if not given directly (useful for training).

    targets_getter : callable, optional
        Get the context indices if not given directly (useful for training).

    is_add_cntxts_to_trgts : bool, optional
        Whether to add the context points to the targets.
    """

    def __init__(
        self,
        contexts_getter=GetRandomIndcs(),
        targets_getter=get_all_indcs,
        is_add_cntxts_to_trgts=False,
    ):
        self.contexts_getter = contexts_getter
        self.targets_getter = targets_getter
        self.is_add_cntxts_to_trgts = is_add_cntxts_to_trgts

    def __call__(
        self, X, y=None, context_indcs=None, target_indcs=None, is_return_indcs=False, indp_target=False
    ):
        """
        Parameters
        ----------
        X: tf.constant, size = [batch_size, num_points, x_dim]
            Position features. Values should always be in [-1, 1].

        Y: tf.constant, size = [batch_size, num_points, y_dim]
            Targets.

        context_indcs : np.array, size=[batch_size, n_indcs]
            Indices of the context points. If `None` generates it using
            `contexts_getter(batch_size, num_points)`.

        target_indcs : np.array, size=[batch_size, n_indcs]
            Indices of the target points. If `None` generates it using
            `contexts_getter(batch_size, num_points)`.

        is_return_indcs : bool, optional
            Whether to return X and the selected context and taregt indices, rather
            than the selected `X_cntxt, Y_cntxt, X_trgt, Y_trgt`.
        
        indp_target: bool, optional
            If True, any overlapping context points are removed in the set of target points
        """
        batch_size, num_points = self.getter_inputs(X)

        if context_indcs is None:
            context_indcs = self.contexts_getter(batch_size, num_points)
        if target_indcs is None:
            target_indcs = self.targets_getter(batch_size, num_points)
            if indp_target:
                target_indcs = setdiff(target_indcs, context_indcs)

        if self.is_add_cntxts_to_trgts:
            target_indcs = self.add_cntxts_to_trgts(
                num_points, target_indcs, context_indcs
            )

        # only used if X for context and target should be different (besides selecting indices!)
        X_pre_cntxt = self.preprocess_context(X)

        if is_return_indcs:
            # instead of features return indices / masks, and `Y_cntxt` is replaced
            # with all values Y
            return (
                context_indcs,
                X_pre_cntxt,
                target_indcs,
                X,
            )
        X_cntxt, Y_cntxt = self.select(X_pre_cntxt, y, context_indcs)
        X_trgt, Y_trgt = self.select(X, y, target_indcs)
        return X_cntxt, Y_cntxt, X_trgt, Y_trgt

    def preprocess_context(self, X):
        """Preprocess the data for the context set."""
        return X

    def add_cntxts_to_trgts(self, num_points, target_indcs, context_indcs):
        """
        Add context points to targets. This might results in duplicate indices in
        the targets.
        """
        #target_indcs = torch.cat([target_indcs, context_indcs], dim=-1)
        target_indcs = tf.concat([target_indcs, context_indcs], dim=-1)
        # to reduce the probability of duplicating indices remove context indices
        # that made target indices larger than n_possible_points
        return target_indcs[:, :num_points]

    def getter_inputs(self, X):
        """Make the input for the getters."""
        batch_size, num_points, x_dim = X.shape
        return batch_size, num_points

    def select(self, X, y, indcs):
        """Select the correct values from X."""
        indcs = convert_indices(indcs)
        return (tf.gather_nd(X, indcs), tf.gather_nd(y, indcs))

###########################################################################################
# Tools to get datasets
###########################################################################################
def get_all_gp_datasets(dataset_split="train", train_datasets=None, n_samples=50000, n_points=128, is_reuse_across_epochs=False, **kwargs):
    """Return train / tets / valid sets for all GP experiments."""
    datasets = dict()

    if dataset_split == "train" or train_datasets is None or dataset_split == MetaSplit.TRAIN or dataset_split == MetaSplit.TRIAL or dataset_split == "any":
        for f in [
            get_datasets_single_gp,
            get_datasets_variable_hyp_gp,
            get_datasets_variable_kernel_gp,
        ]:
            _datasets = f(dataset_split=dataset_split, train_datasets=train_datasets, n_samples=n_samples, n_points=n_points, is_reuse_across_epochs=is_reuse_across_epochs, **kwargs)
            datasets.update(_datasets)

    else:
        for k, dataset in train_datasets.items():
            if dataset.generated_from == "single_gp":
                datasets.update(get_datasets_single_gp(dataset_split=dataset_split, train_datasets={k:dataset}, n_samples=n_samples, n_points=n_points, is_reuse_across_epochs=is_reuse_across_epochs, **kwargs))
            elif dataset.generated_from == "variable_hyp_gp":
                datasets.update(get_datasets_variable_hyp_gp(dataset_split=dataset_split, train_datasets={k:dataset}, n_samples=n_samples, n_points=n_points, is_reuse_across_epochs=is_reuse_across_epochs, **kwargs))
            elif dataset.generated_from == "variable_kernel_gp":
                datasets.update(get_datasets_variable_kernel_gp(dataset_split=dataset_split, train_datasets={k:dataset}, n_samples=n_samples, n_points=n_points, is_reuse_across_epochs=is_reuse_across_epochs, **kwargs))
            else:
                raise Exception("{} in train_datasets are not generated from one of get_datasets_single_gp(), get_datasets_variable_hyp_gp, get_datasets_variable_kernel_gp()".format(dataset))

    return datasets


def get_datasets_single_gp(dataset_split="train", train_datasets=None, kernels=None, n_samples=50000, n_points=128, is_reuse_across_epochs=False, **kwargs):
    """Return train / tets / valid sets for 'Samples from a single GP'."""

    if kernels is None:
        kernels = dict()

        kernels["RBF_Kernel"] = RBF(length_scale=(0.2))

        kernels["Periodic_Kernel"] = ExpSineSquared(length_scale=0.5, periodicity=0.5)

        # kernels["Matern_Kernel"] = Matern(length_scale=0.2, nu=1.5)

        kernels["Noisy_Matern_Kernel"] = WhiteKernel(noise_level=0.1) + Matern(
            length_scale=0.2, nu=1.5
        )

    return get_gp_datasets(
        kernels,
        dataset_split=dataset_split, 
        train_datasets=train_datasets,
        is_vary_kernel_hyp=False,  # use a single hyperparameter per kernel
        train_n_samples=n_samples,  # number of different context-target sets
        val_n_samples=n_samples,
        test_n_samples=n_samples,
        n_points=n_points,  # size of target U context set for each sample
        is_reuse_across_epochs=is_reuse_across_epochs,  # never see the same example twice
        generated_from = "single_gp",
        **kwargs,
    )


def get_datasets_variable_hyp_gp(dataset_split="train", train_datasets=None, kernels=None, n_samples=50000, n_points=128, is_reuse_across_epochs=False, **kwargs):
    """Return train / tets / valid sets for 'Samples from GPs with varying Kernel hyperparameters'."""
    if kernels is None:
        kernels = dict()

        kernels["Variable_Matern_Kernel"] = Matern(length_scale_bounds=(0.01, 0.3), nu=1.5)

    return get_gp_datasets(
        kernels,
        dataset_split=dataset_split, 
        train_datasets=train_datasets,
        is_vary_kernel_hyp=True,  # use a different hyp for each samples
        train_n_samples=n_samples,  # number of different context-target sets
        val_n_samples=n_samples,
        test_n_samples=n_samples,
        n_points=n_points,  # size of target U context set for each sample
        is_reuse_across_epochs=is_reuse_across_epochs,  # never see the same example twice
        generated_from = "variable_hyp_gp",
        **kwargs,
    )


def get_datasets_variable_kernel_gp(dataset_split="train", train_datasets=None, kernels=None, n_samples=50000, n_points=128, is_reuse_across_epochs=False, **kwargs): #default not reuse across epoch
    """Return train / tets / valid sets for 'Samples from GPs with varying Kernels'.
    train_datasets: DatasetMerger
    """

    if dataset_split == "train" or train_datasets is None or dataset_split == MetaSplit.TRAIN or dataset_split == MetaSplit.TRIAL or dataset_split == "any":
        datasets = get_datasets_single_gp(dataset_split=dataset_split, train_datasets=train_datasets, kernels=kernels, n_samples=n_samples, n_points=n_points, is_reuse_across_epochs=is_reuse_across_epochs, **kwargs)
        all_kernels_datasets = DatasetMerger(datasets)
    else:
        train_datasets = list(train_datasets.values())[0]
        if not isinstance(train_datasets, DatasetMerger):
            raise ValueError("train_datasets must be a DatasetMerger")
        datasets = dict()
        for k, dataset in list(zip(train_datasets.datasets_names, train_datasets.datasets)): # get the datasets contained in train_datasets
            datasets.update(get_datasets_single_gp(dataset_split=dataset_split, train_datasets={k:dataset}, kernels=None, n_samples=n_samples, n_points=n_points, is_reuse_across_epochs=is_reuse_across_epochs, **kwargs)) #kernels are not needed when recovering datasets
        all_kernels_datasets = DatasetMerger(datasets)

    all_kernels_datasets.generated_from = "variable_kernel_gp"
    return dict(All_Kernels=all_kernels_datasets) # combine all kernels

def sample_gp_dataset_like(dataset, **kwargs):
    """Wrap the output of `get_samples` in a gp dataset."""
    new_dataset = copy.deepcopy(dataset)
    new_dataset.set_samples_(*dataset.get_samples(**kwargs))
    return new_dataset

def get_gp_datasets(
    kernels=None, save_file=f"{os.path.join(DIR_DATA, 'gp_dataset.hdf5')}", dataset_split="train", 
    train_datasets=None, train_n_samples=50000, val_n_samples=0.1, test_n_samples=10000, **kwargs
):
    """
    Return a train, test or validation set for all the given kernels (dict).

    kernels: dict of sklearn.gaussian_process.kernel kernels
        - kernels for the datasets
    save_file: str
        - path of hdf5 file used to save previously generated datasets
    dataset_split: str
        - one of "train", "val", "test"
    train_datasets: None or GPDataset instance
        - If None, a new validation/test dataset without fixed samples are generated. If GPDataset is input, a new validation/test dataset with the same structure of the supplied dataset is genereated with fixed samples
    train_n_samples: int
        - Number of samples per training dataset
    val_n_samples: int or float
        - Number of samples per validation dataset. If [0,1] is given, it is the proportion of the train_datasets provided
    test_n_samples: int or float
        - Number of samples per validation dataset. If [0,1] is given, it is the proportion of the train_datasets provided
    n_points: int
        - Number of points at which to evaluate f(x) for x in min_max in GPDataset for training dataset. The n_points for validation dataset and test datasets is referred from train_datasets provided.
    is_reuse_across_epochs: bool
        - Whether to reuse the batches for each epoch for training dataset. For validation dataset and test dataset, this parameter is inferred from the train_datasets provided
    """

    if train_datasets is None and kernels is None:
        raise Exception("kernels must be specified when train_datasets is None")

    if dataset_split == "train" or train_datasets is None or dataset_split == MetaSplit.TRAIN or dataset_split == MetaSplit.TRIAL or dataset_split == "any":
        datasets = dict() # store train datasets
        for name, kernel in kernels.items():
            datasets[name] = GPDataset(
                kernel=kernel, save_file=save_file, n_samples=train_n_samples, kernel_name=name, **kwargs
            )

    elif dataset_split == "test" or dataset_split == MetaSplit.TEST:
        # get validation and test datasets
        datasets = { # store test datasets
            k: sample_gp_dataset_like(
                dataset, save_file=save_file, idx_chunk=-1, n_samples=ratio_to_int2(test_n_samples, dataset.n_samples)
            )
            for k, dataset in train_datasets.items()
        }

    elif dataset_split == "valid" or dataset_split == MetaSplit.VALID:
        datasets = { # store val datasets
            k: sample_gp_dataset_like(
                dataset,
                save_file=save_file,
                idx_chunk=-2,
                n_samples=ratio_to_int2(val_n_samples, dataset.n_samples),
            )
            for k, dataset in train_datasets.items()
        }

    else:
        raise NameError("Unknown dataset_split")

    return datasets



if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils import parse_config
    config = parse_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/debug.yaml"))
    dataloader = DataProvider(config)

    dat_gen = dataloader.generate()
    dat = dat_gen[0]["All_Kernels"]
    dat_test = dataloader.generate_test()
    dat_test = dat_test["All_Kernels"]

    for i in dat.take(1):
        pass

    for i in dat_test.take(1):
        pass

    # dat = dat.batch(5, )
    #dat = dat.map(lambda X, y: tf.py_function(getter(X=X, y=y), [X,y], [tf.float32,tf.float32,tf.float32,tf.float32]))

    # map_fun = lambda X, y: getter(X=X, y=y, context_indcs=tf.constant(np.ones([5,10]), dtype=tf.int32), target_indcs=tf.constant(np.ones([5,10]), dtype=tf.int32))
    # dat = dat.map(lambda X, y: tf.py_function(
    #     # func=getter(
    #     #     X=X, y=y, 
    #     #     context_indcs=tf.constant(np.ones([5,10]), dtype=tf.int32), 
    #     #     target_indcs=tf.constant(np.ones([5,10]), dtype=tf.int32)), 
    #     func=map_fun, 
    #     inp=[X,y], 
    #     Tout=[tf.float32,tf.float32,tf.float32,tf.float32]))
    # for i in dat.take(1):
    #     print("hi", i)