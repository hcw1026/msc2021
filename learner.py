#from msc2021.data.tools import RegressionDescription
from math import ceil
from numpy.lib.index_tricks import _fill_diagonal_dispatcher

from tensorflow.python.ops.gen_array_ops import deep_copy
from model import MetaFunClassifier, MetaFunRegressor
import tensorflow as tf
import sonnet as snt
from data.tools import ClassificationDescription, RegressionDescription
from data.gp_regression import DataProvider as gp_provider
from data.leo_imagenet import DataProvider as imagenet_provider
import utils
import os
import numpy as np
from tqdm import tqdm

from datetime import datetime

class BaseLearner():

    def __init__(self, config, model, name="CLearner", model_name=None):
        self.eval_metric_type = "acc"

        self._float_dtype = tf.float32
        self._int_dtype = tf.int32
        
        self.config = config
        self.model = model
        self.model_dupl = model
        self.name = name
        self.model_name = model_name

        # Model Configurations
        self._l2_penalty_weight = config["Model"]["reg"]["l2_penalty_weight"]
        self._use_gradient = config["Model"]["comp"]["use_gradient"]

        # Training Configurations
        _config = config["Train"]
        self._outer_lr = _config["lr"]
        self._train_on_val = _config["train_on_val"]
        self._train_batch_size = tf.constant(_config["batch_size"], dtype=self._float_dtype)
        self._epoch = _config["epoch"]
        self._train_num_per_epoch = tf.constant(_config["num_per_epoch"], dtype=tf.int32) if _config["num_per_epoch"] is not None else tf.constant(999999, dtype=tf.int32)
        self._train_drop_remainder = _config["drop_remainder"]
        self._print_freq = _config["print_freq"]

        self._train_num_takes = ceil(int(self._train_num_per_epoch)/int(self._train_batch_size))

        # Early Stopping Configurations
        _config = config["Train"]["early_stop"]
        self._early_stop = _config["early_stop"]
        self._early_stop_reset = _config["early_stop_reset"]
        self._early_stop_patience = _config["early_stop_patience"]
        self._early_stop_min_delta = abs(_config["early_stop_min_delta"])
        self._early_stop_monitor = _config["early_stop_monitor"]

        self.early_stop_curr_best = tf.Variable(0., dtype=self._float_dtype)
        self.stop = False

        # Checkpoint Configurations
        _config = config["Train"]["save"]
        self._ckpt_save_dir = _config["ckpt_save_dir"]
        self._ckpt_save_prefix = _config["ckpt_save_prefix"]
        self._restore_from_ckpt = _config["restore_from_ckpt"]
        self._ckpt_restore_path = _config["ckpt_restore_path"]
        self._save_final_model = _config["save_final_model"]

        self.best_epoch = tf.Variable(1)

        # Evaluation Configurations
        _config = config["Eval"]
        self._val_batch_size = tf.constant(_config["batch_size"], dtype=self._float_dtype)
        self._validation = _config["validation"]

        self._val_num_per_epoch = tf.constant(_config["num_per_epoch"], dtype=tf.int32) if _config["num_per_epoch"] is not None else tf.constant(999999, dtype=tf.int32)
        self._val_drop_remainder = _config["drop_remainder"]
        self._val_num_takes = ceil(int(self._val_num_per_epoch)/int(self._val_batch_size))

        self._test_batch_size = self._val_batch_size


        # GPU Configurations
        self._gpu = config["GPU"]

        # Load data
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Initialise
        self.train_dist_ds = None
        self.val_dist_ds = None

        self.epoch_counter = tf.Variable(1, trainable=False) #count epoch
        self.current_state = "train"

        self.current_time = datetime.now().strftime("%y%m%d-%H%M%S")

        # Other
        self.signature = None #tf.function input signature
        self.description = ClassificationDescription
        self.has_train = False

        # Setup (multi-)GPU training
        if self._gpu is not None:
            if not isinstance(self._gpu, list):
                self._gpu = list(self._gpu) 
            self.strategy = snt.distribute.Replicator(
                ["/device:GPU:{}".format(i) for i in self._gpu])

        else:
            self.strategy = snt.distribute.Replicator()

    def set_signature(self, signature):
        self.signature = signature

    def load_data_from_datasets(self, training=None, val=None, test=None):
        """load custom data"""
        if training is not None:
            self.train_data = training
            self._train_batch_size = next(iter(self.train_data)).tr_input.shape[0]

        if val is not None:
            self.val_data = val
            self._val_batch_size = next(iter(self.val_data)).tr_input.shape[0]

        if test is not None:
            self.test_data = test
            self._test_batch_size = next(iter(self.test_data)).tr_input.shape[0]

        for dataset in [self.train_data, self.val_data, self.test_data]:
            if dataset is not None:
                embedding_dim_x = next(iter(self.train_data)).tr_input.shape[-1]
                embedding_dim_y = next(iter(self.train_data)).tr_output.shape[-1]

                self.signature = (ClassificationDescription(
                tf.TensorSpec(shape=(None, None, embedding_dim_x), dtype=self._float_dtype), 
                tf.TensorSpec(shape=(None, None, embedding_dim_y), dtype=self._int_dtype),
                tf.TensorSpec(shape=(None, None, embedding_dim_x), dtype=self._float_dtype),
                tf.TensorSpec(shape=(None, None, embedding_dim_y), dtype=self._int_dtype)),)
    
                break

    def _initialise(self, model, data):
        model = self._initialise_model(model)
        self.optimiser = tf.keras.optimizers.Adam(learning_rate=self._outer_lr)
        self.regulariser = snt.regularizers.L2(self._l2_penalty_weight)

        # Initialise model
        model.initialise(next(iter(data)))
        return model

    def train(self):
        """train model"""
        print("Number of devices: {}".format(self.strategy.num_replicas_in_sync))
        with self.strategy.scope():
            self.model = self._initialise(model=self.model, data=self.train_data) # initialise model and optimisers

        self._check_validation()
        self.train_dist_ds = self.strategy.experimental_distribute_dataset(self.train_data)
        self.signature = (iter(self.train_dist_ds).element_spec,)
        train_step = self._tf_func_train_step()
        distributed_train_step = self._distributed_step(train_step) # distributed train_step

        if self._validation:
            validation_step = self._tf_func_val_step()
            distributed_val_step = self._distributed_step(validation_step)
            self.val_dist_ds = self.strategy.experimental_distribute_dataset(self.val_data)
        else:
            distributed_val_step = None


        self._define_metrics()
        with self.strategy.scope():
            self._create_restore_checkpoint()
        
        if self.stop is True:
            print("Early stopping criteria has been reached, training is terminated.")
            return

        # Initialise tensorboard
        self._create_summary_writers()

        self._train_loop(distributed_train_step, distributed_val_step)

        # Save model
        if self._save_final_model:
            tf.saved_model.save(self.model, self._ckpt_save_dir)
        self.has_train = True

    def _train_loop(self, train_step, validation_step):
        """training iterations"""
        # Determine number of steps per epoch
        train_last_step, train_remainder = divmod(int(self._train_num_per_epoch), int(self._train_batch_size))
        val_last_step, val_remainder = divmod(int(self._val_num_per_epoch), int(self._val_batch_size))

        epoch_start = int(self.epoch_counter.numpy())

        for epoch in range(epoch_start, self._epoch+1):

            train_iter = iter(self.train_dist_ds)

            #step_num = 0
            self.current_state = "train"
            print("Train>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            for step_num in tqdm(range(1, self._train_num_takes+1)):
            #for train_batch in self.train_dist_ds.take(self._train_num_takes):
                #step_num += 1
                try:
                    train_batch = next(train_iter)
                except StopIteration:
                    break

                if self._train_drop_remainder and step_num == train_last_step:
                    train_batch = utils.trim(data=train_batch, size=train_remainder, description=self.description)

                train_loss,  train_tr_metric, train_val_metric = train_step(train_batch)
                self.metric_train_target_loss(train_loss) # TODO: check correctness of placing it here

                train_tr_metric = tf.reduce_mean(train_tr_metric)
                train_val_metric = tf.reduce_mean(train_val_metric)

                if step_num % self._print_freq == 0:
                    print("Train -- Epoch: {1}/{2}, Step: {3}, target_loss: {4:.3f}, mean_target_loss: {5:.3f}, context_{0}: {6:.3f}, target_{0}: {7:.3f}, mean_context_{0}: {8:.3f}, mean_target_{0}: {9:.3f}".format(
                    self.eval_metric_type, epoch, self._epoch, step_num, train_loss, self.metric_train_target_loss.result(), train_tr_metric, train_val_metric, self.metric_train_context_metric.result(), self.metric_train_target_metric.result()))

                self._write_tensorboard_step(self.optimiser.iterations, train_loss)

            self._write_tensorboard_epoch(epoch)

            if self._validation:
                print("Validation>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

                val_iter = iter(self.val_dist_ds)

                #step_num = 0
                self.current_state = "val"

                for step_num in tqdm(range(1, self._val_num_takes+1)):
                #for val_batch in self.val_data.take(self._val_num_takes):
                    #step_num += 1
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        break

                    if self._val_drop_remainder and step_num == val_last_step:
                        val_batch = utils.trim(data=val_batch, size=val_remainder, description=self.description)

                    val_loss, val_tr_metric, val_val_metric = validation_step(val_batch)
                    self.metric_val_target_loss(val_loss) # TODO: check correctness of placing it here

                    val_tr_metric = tf.reduce_mean(val_tr_metric)
                    val_val_metric = tf.reduce_mean(val_val_metric)

                    if step_num % self._print_freq == 0:
                        print("Validation --Epoch: {1}/{2}, Step: {3}, target_loss: {4:.3f}, mean_target_loss: {5:.3f}, context_{0}: {6:.3f}, target_{0}: {7:.3f}, mean_context_{0}: {8:.3f}, mean_target_{0}: {9:.3f}".format(
                        self.eval_metric_type, epoch, self._epoch, step_num, val_loss, self.metric_val_target_loss.result(), val_tr_metric, val_val_metric, self.metric_val_context_metric.result(), self.metric_val_target_metric.result()))

            self._write_tensorboard_epoch(epoch)

            if self._validation:
                self._early_stopping()

            self._reset_metrics()
            self._save_checkpoint()
            if self.stop is True:
                print("Early stopping criteria is reached, training is terminated.")
                return
            self.epoch_counter.assign_add(tf.constant(1))

    def _tf_func_train_step(self):
        """one meta-training loop"""

        #@utils.conditional_tf_function(condition= not self._use_gradient, input_signature=self.signature)#@tf.function
        def _train_step(train_batch):
            print("Graph built")
            with tf.GradientTape() as tape:
                train_loss, additional_loss, train_tr_metric, train_val_metric = self.model(train_batch, is_training=True)[:4] #additional_loss is orthogonality loss for imagenet datasets
                reg_loss = self.regulariser(self.model.get_regularise_variables)
                train_loss = utils.combine_losses(train_loss, reg_loss + additional_loss, self._train_batch_size)
                

            gradients = tape.gradient(train_loss, self.model.trainable_variables)
            self.optimiser.apply_gradients(zip(gradients, self.model.trainable_variables))

            # Update metrics
            self.metric_train_target_metric(train_val_metric)
            self.metric_train_context_metric(train_tr_metric)

            return train_loss, train_tr_metric, train_val_metric

        return _train_step

    def _tf_func_val_step(self):
        """one meta-validation loop"""

        #@utils.conditional_tf_function(condition= not self._use_gradient, input_signature=self.signature)#@tf.function #TODO: solve the nested tf.function problem when using nested gradient
        def _val_step(val_batch):
            val_loss, additional_loss, val_tr_metric, val_val_metric = self.model(val_batch, is_training=False)[:4]
            reg_loss = self.regulariser(self.model.get_regularise_variables)
            val_loss = utils.combine_losses(val_loss, reg_loss + additional_loss, self._val_batch_size)

            # Update metrics
            self.metric_val_target_metric(val_val_metric)
            self.metric_val_context_metric(val_tr_metric)

            return val_loss, val_tr_metric, val_val_metric

        return _val_step

    def _distributed_step(self, compute_step_fn):

        @utils.conditional_tf_function(condition= not self._use_gradient, input_signature=self.signature)#@tf.function
        def distributed_train_step(batch):
            per_replica_losses = self.strategy.run(compute_step_fn, args=(batch, ))
            return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        return distributed_train_step
    
    def _check_validation(self):
        """check if validation dataset is None"""
        if not self.val_data:
            print("validation is disabled due to no validation dataset was provided")
            self._validation = False

    def _define_metrics(self):
        self.metric_train_target_loss = tf.keras.metrics.Mean(name="train_target_loss")
        with self.strategy.scope():
            self.metric_train_target_metric = tf.keras.metrics.Mean(name="train_target_{}".format(self.eval_metric_type))
            self.metric_train_context_metric = tf.keras.metrics.Mean(name="train_context_{}".format(self.eval_metric_type))

        if self._validation:
            self.metric_val_target_loss = tf.keras.metrics.Mean(name="val_target_loss")
            with self.strategy.scope():
                self.metric_val_target_metric = tf.keras.metrics.Mean(name="val_target_{}".format(self.eval_metric_type))
                self.metric_val_context_metric = tf.keras.metrics.Mean(name="val_context_{}".format(self.eval_metric_type))

    def _reset_metrics(self):
        self.metric_train_target_loss.reset_states()
        self.metric_train_target_metric.reset_states()
        self.metric_train_context_metric.reset_states()

        if self._validation:
            self.metric_val_target_loss.reset_states()
            self.metric_val_target_metric.reset_states()
            self.metric_val_context_metric.reset_states()

    def _create_restore_checkpoint(self):
        # Checkpoints
        if not self._ckpt_save_prefix:
            self._ckpt_save_prefix = "ckpt"

        # Initialise early stopping tracking metrics
        if self._validation:
            self._early_stopping_init()

        ckpt = tf.train.Checkpoint(epoch_counter=self.epoch_counter, model=self.model, optimiser=self.optimiser, best_epoch=self.best_epoch, early_stop_curr_best=self.early_stop_curr_best)

        init_from_start = utils.ckpt_restore(self, ckpt)
        if init_from_start:
            if self._ckpt_save_dir is None:
                self._ckpt_save_dir = os.path.join("./checkpoint", self.current_time)
            else:
                self._ckpt_save_dir = os.path.join(self._ckpt_save_dir,self.current_time)

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self._ckpt_save_dir, max_to_keep=None, checkpoint_name=self._ckpt_save_prefix)

    def _create_summary_writers(self):
        train_log_dir = os.path.join(self._ckpt_save_dir, "logs/summary/train_epoch")
        train_log_dir_batch = os.path.join(self._ckpt_save_dir, "logs/summary/train_batch")
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.train_summary_writer_batch = tf.summary.create_file_writer(train_log_dir_batch)

        if self._validation:
            val_log_dir = os.path.join(self._ckpt_save_dir, "logs/summary/validation_epoch")
            self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    def _write_tensorboard_epoch(self, epoch):
        if self.current_state == "train":
            with self.train_summary_writer.as_default():
                tf.summary.scalar("Mean Train Target loss", self.metric_train_target_loss.result(), step=epoch)
                tf.summary.scalar("Train Target {}".format(self.eval_metric_type), self.metric_train_target_metric.result(), step=epoch)
                tf.summary.scalar("Train Context {}".format(self.eval_metric_type), self.metric_train_context_metric.result(), step=epoch)
                tf.summary.flush()

        elif self.current_state == "val":
            with self.val_summary_writer.as_default():
                tf.summary.scalar("Mean Validation Target loss", self.metric_val_target_loss.result(), step=epoch)
                tf.summary.scalar("Validation Target {}".format(self.eval_metric_type), self.metric_val_target_metric.result(), step=epoch)
                tf.summary.scalar("Validation Context {}".format(self.eval_metric_type), self.metric_val_context_metric.result(), step=epoch)
                tf.summary.flush()

    def _write_tensorboard_step(self, iteration, loss):
        with self.train_summary_writer.as_default():
            tf.summary.scalar("Train Target loss (step)", loss, step=iteration)
            tf.summary.scalar("Mean Train Target loss (step)", self.metric_train_target_loss.result(), step=iteration)
            tf.summary.scalar("Train Target {} (step)".format(self.eval_metric_type), self.metric_train_target_metric.result(), step=iteration)
            tf.summary.scalar("Train Context {} (step)".format(self.eval_metric_type), self.metric_train_context_metric.result(), step=iteration)
            tf.summary.flush()

    def _early_stopping_init(self):
        # Early stopping output
        if isinstance(self._early_stop_monitor, str):
            self._early_stop_monitor = self._early_stop_monitor.lower()
            if self._early_stop_monitor[-4:] == "loss":
                self.early_stop_curr_best.assign(tf.constant(float("inf")))
            else:
                self.early_stop_curr_best.assign(tf.constant(-float("inf")))

        self.best_epoch.assign(self.epoch_counter.value())

        # Early stopping utils
        self.early_stop_counter = tf.constant(0)
        self.stop = False
        self.early_stop_map = {"loss":self.metric_val_target_loss, "metric":self.metric_val_target_metric}

    def _early_stopping(self):
        curr_metric = self.early_stop_map[self._early_stop_monitor].result()

        # Stopping condition
        if self._early_stop_monitor[-4:] == "loss":
            diff = self.early_stop_curr_best -curr_metric
        else:
            diff = curr_metric - self.early_stop_curr_best
        if diff < self._early_stop_min_delta:
            self.early_stop_counter = self.early_stop_counter + tf.constant(1)
            if self.early_stop_counter >= self._early_stop_patience:
                if self._early_stop:
                    self.stop = True
        else:
            self.early_stop_curr_best.assign(curr_metric)
            self.best_epoch.assign(self.epoch_counter.value())
            self.early_stop_counter = tf.constant(0)

    def _save_checkpoint(self):
        # Save checkpoint
        path = self.ckpt_manager.save(checkpoint_number=self.epoch_counter)
        print("checkpoint is saved to ",path)

        #save config
        dest_dir = os.path.join(self._ckpt_save_dir, "config")
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        utils.save_as_yaml(self.config, os.path.join(dest_dir,"config_epoch_{}.yaml".format(self.epoch_counter.value())))

    def profile(self, with_graph=False, profile_dir=None):
        utils.profile(self, with_graph=with_graph, profile_dir=profile_dir)

    def _test(self, test_size=None, checkpoint_path=None, use_exact_ckpt_path=False, result_save_dir=None, result_save_filename=None, **kwargs):
        """
        test_size: int or None
            - The number of testing samples from the learner's loaded test dataset. If None, all data are used (<=999999)
        checkpoint_path: str or None
            - If not None, the checkpoint will be restored (for this test function only). If the .train method has not been used by the learner, data_source must be provided. If None, the checkpoints from the current checkpoint saving directory in the class will be used
        use_exact_ckpt_path: bool
            - If True, use the newest checkpoint available, otherwise use the checkpoint with the best metric
        result_save_dir: str or None
            - The directory to save the npz results. If None, the result is saved in a subdirectory in the checkpoint path
        result_save_filename: str or None
            - The filename of the save. If None, the filename is "datetime_test_result.npz"
        kwargs:
            - for GPRegressor:
                - save_pred: If True, prediction of each dataset is saved
                - save_data: If True, test data is saved
        """

        if checkpoint_path is not None: # if checkpoint_path is provided, get the current model instance
            if self.has_train:
                with self.strategy.scope():
                    model_instance = self._initialise_model(model=self.model_dupl, data=self.test_data)
            else:
                with self.strategy.scope():
                    model_instance = self._initialise(model=self.model_dupl, data=self.test_data)
                #model_instance = self._initialise_model(model=self.model_dupl)
        else: # if checkpoint_path is not provided, determine if using exact checkpoint path
            if use_exact_ckpt_path: # does not require checkpoint restore
                if self.has_train:
                    model_init = _fill_diagonal_dispatcher
                    model_instance = self.model           
                else:
                    raise Exception("The model has not been trained and no checkpoint_path is given")
            else:
                with self.strategy.scope():
                    model_instance = self._initialise(model=self.model_dupl, data=self.test_data)
                #model_instance = self._initialise_model(model=self.model_dupl)
                checkpoint_path = self._ckpt_save_dir      

        if result_save_dir is None:
            result_save_dir = os.path.join(self._ckpt_save_dir, "test_result")
            if not os.path.isdir(result_save_dir):
                os.mkdir(result_save_dir)

        utils.test(checkpoint_path=checkpoint_path, testloop=self._testloop, model_instance=model_instance, test_data=self.test_data, test_size=test_size, result_save_dir=result_save_dir, result_save_filename=result_save_filename, use_exact_ckpt_path=use_exact_ckpt_path, **kwargs)

class ImageNetLearner(BaseLearner):
    def __init__(self, config, model, data_source="leo_imagenet", model_name="MetaFunClassifier", name="ImageNetLearner"):

        super(ImageNetLearner, self).__init__(config=config, model=model, model_name=model_name, name=name)
        self.data_source = data_source

    def load_data_from_provider(self, dataprovider=imagenet_provider):
        """load data from dataprovider"""
        self.train_data = dataprovider("train", self.config).generate()
        self.val_data = dataprovider("val", self.config).generate()
        self.test_data = dataprovider("test", self.config).generate()

        embedding_dim_x = next(iter(self.train_data)).tr_input.shape[-1]
        embedding_dim_y = next(iter(self.train_data)).tr_output.shape[-1]
        self.signature = (ClassificationDescription(
        tf.TensorSpec(shape=(None, None, embedding_dim_x), dtype=self._float_dtype), 
        tf.TensorSpec(shape=(None, None, embedding_dim_y), dtype=self._int_dtype),
        tf.TensorSpec(shape=(None, None, embedding_dim_x), dtype=self._float_dtype),
        tf.TensorSpec(shape=(None, None, embedding_dim_y), dtype=self._int_dtype)),)

    def _initialise_model(self, model):
        return model(config=self.config, data_source=self.data_source, name=self.model_name)

    def _testloop(self, model_instance, test_data, test_size):

        print("Testing>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # Define test step
        def _test_step(test_batch):
            test_loss, additional_loss, test_tr_metric, test_val_metric = model_instance(test_batch, is_training=False)
            reg_loss = self.regulariser(model_instance.get_regularise_variables)
            test_loss = utils.combine_losses(test_loss, reg_loss + additional_loss, self._test_batch_size)

            # Update metrics
            metric_test_target_metric(test_val_metric)
            metric_test_context_metric(test_tr_metric)

            return test_loss, test_tr_metric, test_val_metric

        # Define metrics
        metric_test_target_loss = tf.keras.metrics.Mean(name="train_target_loss")
        with self.strategy.scope():
            metric_test_target_metric = tf.keras.metrics.Mean(name="train_test_{}".format(self.eval_metric_type))
            metric_test_context_metric = tf.keras.metrics.Mean(name="train_test_{}".format(self.eval_metric_type))

        # Strategy setup
        test_dist_ds = self.strategy.experimental_distribute_dataset(test_data)
        self.signature = (iter(test_dist_ds).element_spec,)
        distributed_test_step = self._distributed_step(_test_step)

        # Looping setup
        test_num_takes = ceil(int(test_size)/int(self._test_batch_size))
        test_last_step, test_remainder = divmod(int(test_size), int(self._test_batch_size))
        test_iter = iter(test_dist_ds)
        test_loss_ls, test_tr_metric_ls, test_val_metric_ls = [], [], []

        # Looping
        for step_num in tqdm(range(1, test_num_takes+1)):

            try:
                test_batch = next(test_iter)
            except StopIteration:
                break

            if step_num == test_last_step:
                test_batch = utils.trim(data=test_batch, size=test_remainder, description=self.description)

            test_loss, test_tr_metric, test_val_metric = distributed_test_step(test_batch)

            metric_test_target_loss(test_loss)
            test_tr_metric = tf.reduce_mean(test_tr_metric)
            test_val_metric = tf.reduce_mean(test_val_metric)

            test_loss_ls.append(test_loss.numpy())
            test_tr_metric_ls.append(test_tr_metric.numpy())
            test_val_metric_ls.append(test_val_metric.numpy())

        # Concatenate result
        test_loss_ls = np.array(test_loss_ls)
        test_tr_metric_ls = np.array(test_tr_metric_ls)
        test_val_metric_ls = np.array(test_val_metric_ls)

        return {
            "test_loss":test_loss_ls, 
            "test_tr_{}".format(self.eval_metric_type):test_tr_metric_ls, 
            "test_val_{}".format(self.eval_metric_type):test_val_metric_ls,
            "mean_test_target_loss":metric_test_target_loss.result().numpy(),
            "mean_test_target_{}".format(self.eval_metric_type):metric_test_target_metric.result().numpy(),
            "mean_test_context_{}".format(self.eval_metric_type):metric_test_context_metric.result().numpy()
            }

    def test(self, test_size=None, checkpoint_path=None, use_exact_ckpt_path=False, result_save_dir=None, result_save_filename=None):
        self._test(test_size=test_size, checkpoint_path=checkpoint_path, use_exact_ckpt_path=use_exact_ckpt_path, result_save_dir=result_save_dir, result_save_filename=result_save_filename)


class GPLearner(BaseLearner):
    def __init__(self, config, model=MetaFunRegressor, model_name="MetaFunRegressor", name="GPLearner"):
        """Gaussain Process 1D function regression - Note that data is needed to be load manually
        config: dict
            - configuration file - the same format as a parsed yaml in the config/sample.yaml
        model: MetaFunRegressor
            - the model for training
        model_name: str
            - name of the model
        name: str
            - name of the learner
        
        """
        super(GPLearner, self).__init__(config=config, model=model, model_name=model_name, name=name)
        self.signature = (RegressionDescription(
        tf.TensorSpec(shape=(None, None, 1), dtype=self._float_dtype), 
        tf.TensorSpec(shape=(None, None, 1), dtype=self._float_dtype),
        tf.TensorSpec(shape=(None, None, 1), dtype=self._float_dtype),
        tf.TensorSpec(shape=(None, None, 1), dtype=self._float_dtype)),)

        self.description = RegressionDescription

        self.eval_metric_type = "mse"

    def load_data_from_provider(self, dataprovider=gp_provider, kernel_name=None, load_type="all", custom_kernels=None, custom_kernels_merge=False):
        """load data from dataprovider"""
        provider = dataprovider(self.config, load_type=load_type, custom_kernels=custom_kernels, custom_kernels_merge=custom_kernels_merge)
        self.train_data, self.test_data = provider.generate(return_valid=False, return_test=True)
        self.val_data = self.train_data
        self.train_data = self.train_data[kernel_name] if kernel_name is not None else list(self.train_data.values())[0]
        self.val_data = self.val_data[kernel_name] if kernel_name is not None else list(self.val_data.values())[0]
        self.test_data = self.test_data[kernel_name] if kernel_name is not None else list(self.test_data.values())[0]

    def load_data_from_datasets(self, training=None, val=None, test=None):
        """load custom data"""
        self.train_data = training if training is not None else self.train_data
        self.val_data = val if val is not None else self.val_data
        self.test_data = test if test is not None else self.test_data
    
    def _initialise_model(self, model):
        return model(config=self.config, name=self.model_name)

    def _testloop(self, model_instance, test_data, test_size, save_pred=False, save_data=False):

        print("Testing>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # Define test step
        def _test_step(test_batch):
            test_loss, additional_loss, test_tr_metric, test_val_metric, all_val_mu, all_val_sigma, all_tr_mu, all_tr_sigma = model_instance(test_batch, is_training=False)
            reg_loss = self.regulariser(model_instance.get_regularise_variables)
            test_loss = utils.combine_losses(test_loss, reg_loss + additional_loss, self._test_batch_size)

            # Update metrics
            metric_test_target_metric(test_val_metric)
            metric_test_context_metric(test_tr_metric)

            return test_loss, test_tr_metric, test_val_metric, all_val_mu, all_val_sigma, all_tr_mu, all_tr_sigma

        # Define metrics
        metric_test_target_loss = tf.keras.metrics.Mean(name="train_target_loss")
        with self.strategy.scope():
            metric_test_target_metric = tf.keras.metrics.Mean(name="train_test_{}".format(self.eval_metric_type))
            metric_test_context_metric = tf.keras.metrics.Mean(name="train_test_{}".format(self.eval_metric_type))

        # Strategy setup
        test_dist_ds = self.strategy.experimental_distribute_dataset(test_data)
        self.signature = (iter(test_dist_ds).element_spec,)
        distributed_test_step = self._distributed_step(_test_step)

        # Looping setup
        test_num_takes = ceil(int(test_size)/int(self._test_batch_size))
        test_last_step, test_remainder = divmod(int(test_size), int(self._test_batch_size))
        test_iter = iter(test_dist_ds)
        test_loss_ls, test_tr_metric_ls, test_val_metric_ls = [], [], []
        if save_pred:
            tr_mu_ls, tr_sigma_ls, val_mu_ls, val_sigma_ls = [], [], [], []
        if save_data:
            tr_input_ls, tr_output_ls, val_input_ls, val_output_ls = [], [], [], []

        # Looping
        for step_num in tqdm(range(1, test_num_takes+1)):

            try:
                test_batch = next(test_iter)
            except StopIteration:
                break

            if step_num == test_last_step:
                test_batch = utils.trim(data=test_batch, size=test_remainder, description=self.description)

            test_loss, test_tr_metric, test_val_metric, all_val_mu, all_val_sigma, all_tr_mu, all_tr_sigma = distributed_test_step(test_batch)
            
            metric_test_target_loss(test_loss)
            test_tr_metric = tf.reduce_mean(test_tr_metric)
            test_val_metric = tf.reduce_mean(test_val_metric)

            test_loss_ls.append(test_loss.numpy())
            test_tr_metric_ls.append(test_tr_metric.numpy())
            test_val_metric_ls.append(test_val_metric.numpy())

            if save_pred:
                tr_mu_ls.append(all_tr_mu.numpy().tolist())
                tr_sigma_ls.append(all_tr_sigma.numpy().tolist())
                val_mu_ls.append(all_val_mu.numpy().tolist())
                val_sigma_ls.append(all_val_sigma.numpy().tolist())
            
            if save_data:
                tr_input_ls.append(test_batch.tr_input.numpy().tolist())
                tr_output_ls.append(test_batch.tr_output.numpy().tolist())
                val_input_ls.append(test_batch.val_input.numpy().tolist())
                val_output_ls.append(test_batch.val_output.numpy().tolist())

        # Concatenate result
        test_loss_ls = np.array(test_loss_ls)
        test_tr_metric_ls = np.array(test_tr_metric_ls)
        test_val_metric_ls = np.array(test_val_metric_ls)

        if save_pred:
            tr_mu_ls = np.array(tr_mu_ls, dtype=object)
            tr_sigma_ls = np.array(tr_sigma_ls, dtype=object)
            val_mu_ls = np.array(val_mu_ls, dtype=object)
            val_sigma_ls = np.array(val_sigma_ls, dtype=object)
        
        if save_data:
            tr_input_ls = np.array(tr_input_ls, dtype=object)
            tr_output_ls = np.array(tr_output_ls, dtype=object)
            val_input_ls = np.array(val_input_ls, dtype=object)
            val_output_ls = np.array(val_output_ls, dtype=object)


        output = {
            "test_loss":test_loss_ls, 
            "test_tr_{}".format(self.eval_metric_type):test_tr_metric_ls, 
            "test_val_{}".format(self.eval_metric_type):test_val_metric_ls,
            "mean_test_target_loss":metric_test_target_loss.result().numpy(),
            "mean_test_target_{}".format(self.eval_metric_type):metric_test_target_metric.result().numpy(),
            "mean_test_context_{}".format(self.eval_metric_type):metric_test_context_metric.result().numpy()
            }

        if save_pred:
            output["tr_mu"] = tr_mu_ls
            output["tr_sigma"] = tr_sigma_ls
            output["val_mu"] = val_mu_ls
            output["val_sigma"] = val_sigma_ls
        
        if save_data:
            output["tr_input"] = tr_input_ls
            output["tr_output"] = tr_output_ls
            output["val_input"] = val_input_ls
            output["val_output"] = val_output_ls

        return output

    def test(self, test_size=None, checkpoint_path=None, use_exact_ckpt_path=False, result_save_dir=None, result_save_filename=None, save_pred=False, save_data=False):
        self._test(test_size=test_size, checkpoint_path=checkpoint_path, use_exact_ckpt_path=use_exact_ckpt_path, result_save_dir=result_save_dir, result_save_filename=result_save_filename, save_pred=save_pred, save_data=save_data)


if __name__ == "__main__":
    from utils import parse_config
    from model import MetaFunClassifier
    import os
    import numpy as np
    import collections
    config = parse_config(os.path.join(os.path.dirname(__file__),"config/debug.yaml"))
    from data.leo_imagenet import DataProvider as imagenet_provider

    # mylearner = ImageNetLearner(config, MetaFunClassifier, data_source="leo_imagenet")
    # mylearner.load_data_from_provider(dataprovider=imagenet_provider)
    # mylearner.train()


    from data.gp_regression import DataProvider as gp_provider
    mylearn2 = GPLearner(config, MetaFunRegressor)
    gp_dataloader = gp_provider(config=config)
    gp_data = gp_dataloader.generate()
    gp_train_data = gp_data[0]["RBF_Kernel"]
    gp_test_data = gp_data[1]["RBF_Kernel"]
    mylearn2.load_data_from_datasets(training=gp_train_data, val=gp_train_data, test=gp_test_data)
    mylearn2.train()


    
