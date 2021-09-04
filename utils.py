import yaml
import os
import shutil
from datetime import datetime
import numpy as np
import glob

import tensorflow as tf

#############################################################################################
# config
#############################################################################################
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

def add_to_yaml(dir, levels, new_level, value, manual_glob=False):
    if not manual_glob:
        paths = glob.glob(os.path.join(dir, "*"))
    else:
        paths = glob.glob(dir)

    for path in paths:
        config = parse_config(path)

        if len(levels) > 0:
            c = config[levels[0]]

            for idx, level in enumerate(levels[1:]):
                c = c[level]
        else:
            c = config

        c[new_level] = value
        save_as_yaml(config, path)        

#############################################################################################
# learner
#############################################################################################

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
                print()
                print("Restored from checkpoint", cls._ckpt_restore_path)
                init_from_start = False

                # Check if need to reset early_stopping
                if (cls._early_stop_reset is True) and cls._validation:
                    cls._early_stopping_init()
                    print()
                    print("early stopping tracking has been reset as required")

                cls.early_stop_counter = cls.epoch_counter - cls.best_epoch

                if (cls.early_stop_counter >= cls._early_stop_patience) and cls._early_stop:
                    cls.stop = True

                cls.epoch_counter.assign_add(tf.constant(1))

                if cls._ckpt_save_dir is None:
                    cls._ckpt_save_dir = os.path.dirname(cls._ckpt_restore_path)
                elif os.path.dirname(cls._ckpt_restore_path) != cls._ckpt_save_dir:
                    print()
                    print("Warning: save directory and restore directory could be different, please abort unless this is intended")

            except AssertionError:
                print("Checkpoint not found, Initialising the model without checkpoint")
                print()
                init_from_start = True
        else:
            print()
            print("Checkpoint not found, initialising from scratch")
            init_from_start = True

    else:
        print()
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
        print()
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

def test(testloop, model_instance, test_data, test_size, current_time=None, checkpoint_path=None, use_exact_ckpt=False, result_save_dir=None, result_save_filename=None, **kwargs):
    """function for testing
    model_instance: MetaFunClassifier/MetaFunRegressor
        - A created model class object
    testloop: func
        - A function which takes test input of the form testloop(model_instance, test_data), and return a dictionary of numpy arrays results to be saved
    test_data: tf.data.Dataset
        - A test dataset to be used by model_instance
    checkpoint_path: str or None
        - The exact path or directory used to find the newest or best epoch checkpoint. If None, the model_instance supplied is used directly
    use_exact_ckpt: bool
        - If True, use the newest checkpoint available, otherwise use the checkpoint with the best metric
    result_save_dir: str or None
        - The directory to save the npz results. If None, the result is saved in a subdirectory in the checkpoint path
    result_save_filename: str or None
        - The filename of the save. If None, the filename is "test_result.npz"
    """

    test_size = tf.constant(test_size, dtype=tf.int32) if test_size is not None else tf.constant(999999, dtype=tf.int32)

    # Initialise Model
    if not model_instance.has_initialised:
        model_instance.initialise(next(iter(test_data)))

    _ = model_instance(next(iter(test_data))) # fully initialise trainable variables

    # Restore
    if checkpoint_path is not None:
        if not use_exact_ckpt:
            best_epoch = tf.Variable(0, trainable=False)

            # Find best epoch if directory is given
            restore_path_pre = find_ckpt_path(checkpoint_path)
            ckpt_pre = tf.train.Checkpoint(best_epoch=best_epoch)
            ckpt_pre.restore(restore_path_pre).expect_partial()
            print()
            print("A checkpoint path is found:", restore_path_pre)
            best_epoch = int(tf.Variable(best_epoch).numpy())

            # Compute best epoch path
            restore_path = restore_path_pre.split("-")[:-1]
            restore_path += [str(best_epoch)]
            restore_path = "-".join(restore_path)

            # Restore model
            ckpt = tf.train.Checkpoint(model=model_instance)
            try:
                rp = ckpt.restore(restore_path).expect_partial()
                print()
                print("Checkpoint with best epoch is found and restored", restore_path)
            except:
                rp = ckpt.restore(restore_path_pre).expect_partial()
                print()
                print("Warning: the checkpoint with best epoch cannot be restored, the checkpoint path found earlier is restored instead", restore_path_pre)
                restore_path = restore_path_pre
        else:
            ckpt = tf.train.Checkpoint(model=model_instance)
            restore_path = checkpoint_path
            ckpt.restore(restore_path).expect_partial()
    else:
        restore_path = None


    result = testloop(model_instance, test_data, test_size, **kwargs) # a function which returns dictionary of tensorflow tensors

    # Write result
    if result_save_filename is not None:
        if result_save_filename[-4:] != ".npz":
            result_save_filename = result_save_filename + ".npz"
    else:
        result_save_filename = current_time + "_test_result.npz"

    if result_save_dir is None:
        if restore_path is None:
            raise Exception("result_save_dir must be provided unless checkpoint_path is not None")
        result_save_dir = os.path.join(os.path.dirname(restore_path), "test_result")

    result_save_path = os.path.join(result_save_dir, result_save_filename)

    if not os.path.isdir(result_save_dir):
        os.mkdir(result_save_dir)

    np.savez(result_save_path, **result)
    print()
    print("result is saved at", result_save_path)

    output_mean_res = dict()
    for key, value in result.items():
        if key.startswith("mean_"):
            output_mean_res[key] = value

    return restore_path, result_save_path, output_mean_res


#############################################################################################
# run.py
#############################################################################################

#https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth/31039095
def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def remove_checkpoints(best_epoch, last_epoch, ckpt_dir, ckpt_prefix):
    paths = glob.glob(os.path.join(ckpt_dir, ckpt_prefix+"*"))
    for path in paths:
        basename = os.path.basename(path)
        ckpt = True if  (os.path.splitext(basename)[1].startswith(".data") or os.path.splitext(basename)[1].startswith(".index")) else False
        ckpt_num = int((os.path.splitext(basename)[0][(len(ckpt_prefix)+1):])) if ckpt else -1
        if ckpt and os.path.isfile(path):
            if (ckpt_num == best_epoch-1) or (ckpt_num == best_epoch) or (ckpt_num == best_epoch+1) or (ckpt_num == last_epoch):
                pass
            else:
                os.remove(path)

def copy_checkpoint(checkpoint_path, dest):
    for path in glob.glob(checkpoint_path + "*"):
        filetype = path.split(".")[-1]
        save_path = os.path.join(dest, "best_ckpt" + "." + filetype)
        shutil.copy2(path, save_path)


#############################################################################################
# model
#############################################################################################
def disable_decoder(model_cls):
    model_cls.no_decoder = True

    def empty_fun():
        pass

    model_cls.predict_init = empty_fun
    model_cls.forward_decoder_base = empty_fun
    model_cls.calculate_loss_and_metrics_init = empty_fun