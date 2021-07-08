import yaml
import os
from datetime import datetime

import tensorflow as tf

def parse_config(yaml_path):
    """convert a yaml path into the corresponding dictionary
    Input:
        - yaml_path: str
            - the path of a yaml config
    """
    loader = yaml.SafeLoader
    
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=loader)

    return config

def save_as_yaml(config, path):
    """ save a dictionary as yaml
    Input:
        - config: dict
        - path: str
    """
    with open(path, "w") as f:
        yaml.dump(config, f)

def get_linear_layer_variables(module):
    """obtain all linear layer variables in a snt.Module
    Input:
        - module: snt.Module
            - the module to get the tf.Variable from
    Output:
        - list of tf.Variables which are the linear weights of a MLP and not bias
    """
    variables = []
    for var in module.trainable_variables:
        layer, weight_bias = var.name.split("/")[-2:]
        if layer.startswith("linear") and weight_bias[0]=="w":
            variables.append(var)
    return variables

def combine_losses(per_example_loss, reg_loss, global_batch_size):
    """combine per example loss with reg loss for distributed training
    Input:
        - per_example_loss: tf.constant (1D)
            - each entry corresponds to a loss value for a data instance
        - reg_loss: tf.constant (0D)
            - regularisation loss
    Output:
        - tf.constant (0D)
    """
    example_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size)
    return example_loss + tf.nn.scale_regularization_loss(reg_loss)

def find_ckpt_path(ckpt_restore_path):
    if os.path.isdir(ckpt_restore_path):
    #find further subdirectory -- for restore_path with timestamp as directory name
        in_dir = os.listdir(ckpt_restore_path)
        in_dir = [i for i in in_dir if os.path.isdir(os.path.join(ckpt_restore_path,i))]

        if len(in_dir) == 0:
            ckpt_restore_path = tf.train.latest_checkpoint(ckpt_restore_path)
        if len(in_dir) == 1: #if contain 1 subdirectory, search outside the subdirectory first
            ckpt_restore_path_tmp = tf.train.latest_checkpoint(ckpt_restore_path)
            if ckpt_restore_path_tmp is None:
                ckpt_restore_path = tf.train.latest_checkpoint(os.path.join(ckpt_restore_path,in_dir[0]))
            else:
                ckpt_restore_path = ckpt_restore_path_tmp
        elif len(in_dir) >1 : #if more than one, raise error unless a checkpoint is found outside the subdirectories
            ckpt_restore_path_tmp = tf.train.latest_checkpoint(ckpt_restore_path)
            if ckpt_restore_path_tmp is None:
                raise AssertionError("More than one subdirectories are found within ckpt_restore_path, abort")
            else:
                ckpt_restore_path = ckpt_restore_path_tmp

    return ckpt_restore_path

def ckpt_restore(cls, ckpt):
    """restore ckpt
    Input: 
        - cls: class instance
        - ckpt: tf.train.Checkpoint
    Output:
        - init_from_start: bool
    
    """

    if cls._restore_from_ckpt is True:
        cls._ckpt_restore_path = find_ckpt_path(cls._ckpt_restore_path)
        if cls._ckpt_restore_path is not None:
            try:
                rp = ckpt.restore(cls._ckpt_restore_path)

                rp.assert_existing_objects_matched()
                print("Restored from checkpoint", cls._ckpt_restore_path)
                init_from_start = False

                # Check if need to reset early_stopping
                if (cls._early_stop_reset is True) and cls._validation:
                    cls._early_stopping_init()
                    print("early stopping tracking has been reset as required")

                cls.early_stop_counter = cls.epoch_counter - cls.best_epoch

                if (cls.early_stop_counter >= cls._early_stop_patience) and cls._early_stop:
                    cls.stop = True

                cls.epoch_counter.assign_add(tf.constant(1))

                if cls._ckpt_save_dir is None:
                    cls._ckpt_save_dir = os.path.dirname(cls._ckpt_restore_path)
                elif os.path.dirname(cls._ckpt_restore_path) != cls._ckpt_save_dir:
                    print("Warning: save directory and restore directory could be different, please abort unless this is intended")

            except AssertionError:
                print("Checkpoint not found, Initialising the model without checkpoint")
                init_from_start = True
        else:
            print("Checkpoint not found, initialising from scratch")
            init_from_start = True

    else:
        print("Initialising the model without checkpoint")
        init_from_start = True

    return init_from_start


def profile(cls, with_graph=False, profile_dir=None):
    
    if profile_dir == None:
        profile_dir = "."

    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    logdir = os.path.join(profile_dir,'logs/func/%s' % stamp)
    prof_writer = tf.summary.create_file_writer(logdir)

    if with_graph is True:
        print("Warning: as graph mode is activated, validation will not be performed, and only one epoch will be run")
        tf.summary.trace_on(graph=True, profiler=True)
        cls._validation = False
        cls._epochs = 1
    else:
        tf.summary.trace_on(graph=False,profiler=True)

    #run train loop
    cls.train()

    with prof_writer.as_default():
        tf.summary.trace_export(name="profiling", step=0, profiler_outdir=logdir)

    tf.summary.trace_off()

    #reset
    cls.__init__(config=cls.config, dataprovider=cls.dataprovider, model=cls.model, load_data=cls.load_data, name=cls.name)

def conditional_tf_function(condition, input_signature=None):
    return tf.function(input_signature=input_signature) if condition else lambda x: x

def trim(data, size, description):
    data = list(data)
    for idx, d in enumerate(data):
        data[idx] = d[:size]
    return description(*data)
