#from msc2021.data.tools import RegressionDescription
from math import ceil
from model import MetaFunClassifier, MetaFunRegressor
import tensorflow as tf
import sonnet as snt
from data.tools import ClassificationDescription, RegressionDescription
import utils
import os

from datetime import datetime

class CLearner():

    def __init__(self, config, model, dataprovider=None, load_data=True, name="CLearner"):
        self.eval_metric_type = "acc"

        self._float_dtype = tf.float32
        self._int_dtype = tf.int32
        
        self.config = config
        self.dataprovider = dataprovider
        self.model = model
        self.load_data = load_data
        self.name = name

        # Model Configurations
        self._l2_penalty_weight = config["Model"]["reg"]["l2_penalty_weight"]
        self._use_gradient = config["Model"]["comp"]["use_gradient"]

        # Training Configurations
        _config = config["Train"]
        self._outer_lr = _config["lr"]
        self._train_on_val = _config["train_on_val"]
        self._train_batch_size = tf.constant(_config["batch_size"], dtype=self._float_dtype)
        self._epoch = _config["epoch"]
        self._train_num_per_epoch = tf.constant(_config["num_per_epoch"], dtype=tf.int32)
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

        self._val_num_per_epoch = tf.constant(_config["num_per_epoch"], dtype=tf.int32)
        self._val_drop_remainder = _config["drop_remainder"]
        self._val_num_takes = ceil(int(self._val_num_per_epoch)/int(self._val_batch_size))


        # GPU Configurations
        self._gpu = config["GPU"]

        # Load data
        self.train_data = None
        self.val_data = None
        self.test_data = None
        if load_data is True:
            self._load_data()

        # Initialise
        self.train_dist_ds = None
        self.val_dist_ds = None

        self.epoch_counter = tf.Variable(1, trainable=False) #count epoch
        self.current_state = "train"

        self.current_time = datetime.now().strftime("%y%m%d-%H%M%S")

        # Other
        self.signature = None #tf.function input signature
        self.description = ClassificationDescription

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

    def _load_data(self):
        """load data from dataprovider"""
        self.train_data = self.dataprovider("train", self.config).generate()
        self.val_data = self.dataprovider("val", self.config).generate()
        self.test_data = self.dataprovider("test", self.config).generate()

        embedding_dim_x = next(iter(self.train_data)).tr_input.shape[-1]
        embedding_dim_y = next(iter(self.train_data)).tr_output.shape[-1]
        self.signature = (ClassificationDescription(
        tf.TensorSpec(shape=(None, None, embedding_dim_x), dtype=self._float_dtype), 
        tf.TensorSpec(shape=(None, None, embedding_dim_y), dtype=self._int_dtype),
        tf.TensorSpec(shape=(None, None, embedding_dim_x), dtype=self._float_dtype),
        tf.TensorSpec(shape=(None, None, embedding_dim_y), dtype=self._int_dtype)),)


    def load_custom_data(self, training=None, val=None, test=None):
        """load custom data"""
        self.train_data = training if training is not None else self.train_data
        self.val_data = val if val is not None else self.val_data
        self.test_data = test if test is not None else self.test_data
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

    def train(self, data_source="leo_imagenet", name="MetaFunClassifier"):
        """train model"""

        print("Number of devices: {}".format(self.strategy.num_replicas_in_sync))
        with self.strategy.scope():
            self._initialise_model(data_source=data_source, name=name) # initialise model and optimisers
        self._check_validation()
        train_step = self._tf_func_train_step()
        distributed_train_step = self._distributed_step(train_step) # distributed train_step
        self.train_dist_ds = self.strategy.experimental_distribute_dataset(self.train_data)

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


    def _initialise_model(self, data_source, name):
        self.model = self.model(config=self.config, data_source=data_source, name=name)
        self.optimiser = tf.keras.optimizers.Adam(learning_rate=self._outer_lr)
        self.regulariser = snt.regularizers.L2(self._l2_penalty_weight)

        # Initialise model
        self.model.initialise(next(iter(self.train_data.take(1))))

    def _train_loop(self, train_step, validation_step):
        """training iterations"""
        # Determine number of steps per epoch
        train_last_step, train_remainder = divmod(int(self._train_num_per_epoch), int(self._train_batch_size))
        val_last_step, val_remainder = divmod(int(self._val_num_per_epoch), int(self._val_batch_size))

        epoch_start = int(self.epoch_counter.numpy())

        for epoch in range(epoch_start, self._epoch+1):
            step_num = 0
            self.current_state = "train"
            print("Train>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            for train_batch in self.train_data.take(self._train_num_takes):
                step_num += 1

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
                step_num = 0
                self.current_state = "val"
                for val_batch in self.val_data.take(self._val_num_takes):
                    step_num += 1

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
                self.metric_val_context_metric = tf.keras.metrics.Mean(name="val_target_{}".format(self.eval_metric_type))

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
        utils.save_as_yaml(self.config, os.path.join(dest_dir,"config_epoch_{}.py".format(self.epoch_counter.value())))

    def profile(self, with_graph=False, profile_dir=None):
        utils.profile(self, with_graph=with_graph, profile_dir=profile_dir)



class GPLearner(CLearner):
    def __init__(self, config, model, dataprovider=None, load_data=True, name="GPLearner"):
        super(GPLearner, self).__init__(config=config, model=model, dataprovider=dataprovider, load_data=False, name=name)
        self.signature = (RegressionDescription(
        tf.TensorSpec(shape=(None, None, 1), dtype=self._float_dtype), 
        tf.TensorSpec(shape=(None, None, 1), dtype=self._float_dtype),
        tf.TensorSpec(shape=(None, None, 1), dtype=self._float_dtype),
        tf.TensorSpec(shape=(None, None, 1), dtype=self._float_dtype)),)

        self.description = RegressionDescription

        self.eval_metric_type = "mse"

        if load_data is True:
            self._load_data()

    def _load_data(self):
        """load data from dataprovider"""
        # provider = self.dataprovider("train", self.config)
        # self.train_data = provider.generate()
        # self.val_data = self.train_data
        # self.test_data = provider.generate_test()
        pass


    def load_custom_data(self, training=None, val=None, test=None):
        """load custom data"""
        self.train_data = training if training is not None else self.train_data
        self.val_data = val if val is not None else self.val_data
        self.test_data = test if test is not None else self.test_data



if __name__ == "__main__":
    from utils import parse_config
    from model import MetaFunClassifier
    import os
    import numpy as np
    import collections
    config = parse_config(os.path.join(os.path.dirname(__file__),"config/debug.yaml"))
    from data.leo_imagenet import DataProvider as imagenet_provider

    # mylearner = CLearner(config, MetaFunClassifier, dataprovider=imagenet_provider)
    # mylearner.train()


    from data.gp_regression import DataProvider as gp_provider
    mylearn2 = GPLearner(config, MetaFunRegressor, dataprovider=None, load_data=False)
    gp_dataloader = gp_provider(dataset_split="train", config=config)
    gp_train_data = gp_dataloader.generate()["RBF_Kernel"]
    mylearn2.load_custom_data(training=gp_train_data, val=gp_train_data)
    mylearn2.train()


    
