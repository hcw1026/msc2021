import argparse
import os
import pandas as pd
import random
import tensorflow as tf
import numpy as np
from time import time
from utils import parse_config, copytree, remove_checkpoints
from data.tools import dataset_translation

import experiments

def GPTrain(learner, train_data, val_data, test_data, train=True, extra_data=None, **kwargs):
    """perform GP experiment"""
    learner.load_data_from_datasets(training=train_data, val=val_data, test=test_data, extra=extra_data)
    if train:
        learner.train()
    return learner

def GPTest(learner, test_size=None, checkpoint_path=None, use_exact_ckpt=False, result_save_dir=None, result_save_filename=None, save_pred=False, save_data=False, **kwargs):
    output = learner.test(test_size=test_size, checkpoint_path=checkpoint_path, use_exact_ckpt=use_exact_ckpt, result_save_dir=result_save_dir, result_save_filename=result_save_filename, save_pred=save_pred, save_data=save_data)
    return learner, output

def GPLearnerLoad(learner, config, model, model_name="MetaFunRegressor", name=None, **kwargs):
    return learner(config=config, model=model, model_name=model_name, name=name)

def GPDataLoad(dataprovider, config, load_type, custom_kernels, custom_kernels_merge, **kwargs):
    dataloader = dataprovider(config=config, load_type=load_type, custom_kernels=custom_kernels, custom_kernels_merge=custom_kernels_merge)
    train_data, val_data, test_data = dataloader.generate(return_valid=True, return_test=True, val_is_reuse_across_epochs=False, test_is_reuse_across_epochs=True) # is_reuse_across_epochs follows the convention of the NP processes experiments used
    return dict(train_data=list(train_data.values())[0],
                val_data=list(val_data.values())[0], 
                test_data=list(test_data.values())[0])

def GPDataLoadTE(dataprovider, config, load_type, custom_kernels, custom_kernels_merge, offsets, **kwargs): #offsets must be a list
    train_df, val_df, test_df = GPDataLoad(dataprovider=dataprovider, config=config, load_type=load_type, custom_kernels=custom_kernels, custom_kernels_merge=custom_kernels_merge, **kwargs).values()
    test_df = dataset_translation(test_df, offsets=offsets)
    return dict(train_data=train_df, val_data=val_df, test_data=test_df)

def GPDataLoadGL1(dataprovider, config, load_type, custom_kernels, custom_kernels_merge, **kwargs):
    dataloader = dataprovider(config=config, load_type=load_type, custom_kernels=custom_kernels, custom_kernels_merge=custom_kernels_merge)
    train_data, val_data = dataloader.generate(return_valid=True, return_test=False, val_is_reuse_across_epochs=True)
    dataloader2 = dataprovider(config=config, load_type=load_type, custom_kernels=custom_kernels, custom_kernels_merge=custom_kernels_merge, eval_batch_size=1, min_context=0, max_context=10, min_target=50, max_target=50)
    extra_data = dataloader2.generate_test(n_samples=4096, batch_size=16)
    test_data = dataloader2.generate_test(n_samples=5000, batch_size=1)
    return dict(train_data=list(train_data.values())[0], 
                val_data=list(val_data.values())[0], 
                extra_data=list(extra_data.values())[0], 
                test_data=list(test_data.values())[0])

def ImageNetTrain(learner, train_data, val_data, test_data, train=True, **kwargs):
    """perform imagenet classification experiment"""
    learner.load_data_from_datasets(training=train_data, val=val_data, test=test_data)
    if train:
        learner.train()
    return learner

def ImageNetTest(learner, test_size=None, checkpoint_path=None, use_exact_ckpt=False, result_save_dir=None, result_save_filename=None, **kwargs):
    output = learner.test(test_size=test_size, checkpoint_path=checkpoint_path, use_exact_ckpt=use_exact_ckpt, result_save_dir=result_save_dir, result_save_filename=result_save_filename)
    return learner, output

def ImageNetLearnerLoad(learner, config, model, data_source="leo_imagenet", model_name="MetaFunClassifier", name=None, **kwargs):
    return learner(config=config, model=model, data_source=data_source, model_name=model_name, name=name)

def ImageNetDataLoad(dataprovider, config, **kwargs):
    train_data = dataprovider("train", config).generate()
    val_data = dataprovider("val", config).generate()
    test_data = dataprovider("test", config).generate()
    return train_data, val_data, test_data

# https://stackoverflow.com/questions/60058588/tesnorflow-2-0-tf-random-set-seed-not-working-since-i-am-getting-different-resul
def reset_random_seeds(seed):
   os.environ['PYTHONHASHSEED']=str(seed)
   tf.random.set_seed(2)
   np.random.seed(seed)
   random.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", type=str, help="experiment number - refer to experiments.py", nargs="+")
    parser.add_argument("-i", "--no-train", action="store_true")
    parser.add_argument("-t", "--no-test", action="store_true")

    parser.add_argument("--repeats", type=int, default=1, help="number of repeats for each experiment")
    parser.add_argument("--repeats-start-from", type=int, default=1, help="the repeat number to begin, must be less than --repeat")

    parser.add_argument("--cuda", help="set cuda visible devices", type=int, nargs='+')
    parser.add_argument("--config-dir", default="/data/ziz/chho/msc2021/Code/msc2021/config", help="directory of the configuration files")
    parser.add_argument("--data-source", default="gp_regression", help="the data sublevel of Data in config (for future use only)")
    parser.add_argument("--data-path", default=None, help="path to save or recover gp data, should have suffix .hdf5")

    parser.add_argument("--ckpt-train-save-dir", default=None, help="the directory to save checkpoint, should agree with ckpt-restore-path")
    parser.add_argument("--ckpt-train-restore-path", default=None, help="the directory or exact path to restore checkpoint for training, should agree with ckpt-restore-path")
    parser.add_argument("-r", "--restore-from-checkpoint", action="store_true", help="restore from a checkpoint specified by ckpt-train-restore-path")
    parser.add_argument("-k", "--keep-ckpts", action="store_true", help="By default, checkpoints except the best epoch and +-1 best epoch and the last epoch are removed after each training. This flag keeps all most recent checkpoints (patience+1)")

    parser.add_argument("--test-size", default=None, type=int, help="number of samples for testing, default uses all")
    parser.add_argument("--ckpt-test-restore-path", default=None, help="For testing-only use only. The directory or exact path to restore checkpoint for testing. If None and training is run, restore directly from checkpoints saved in the training")
    parser.add_argument("-e", "--use-exact-ckpt", action="store_true", help="use the exact (or latest) checkpoint in the directory of ckpt-test-restore-path (or from training if ckpt-test-restore-path is None)")
    parser.add_argument("--save-dir", default=None, help="directory to save test results")
    parser.add_argument("--save-filename", default=None, help="filename to save test results")
    parser.add_argument("-p", "--save-pred", action="store_true", help="save predictions (not available for imagenet)") #TODO: implement this for imagenet
    parser.add_argument("-d", "--save-data", action="store_true", help="save testing data (not available for imagenet)")

    parser.add_argument("--train-csv-path", default="/data/ziz/chho/msc2021/Result/training.csv")
    parser.add_argument("--test-csv-path", default="/data/ziz/chho/msc2021/Result/testing.csv")

    parser.add_argument("-c", "--comment", default=None, help="extra comment to add and store")
    parser.add_argument("--debug", action="store_true", help="debugging, for internal use only")
    parser.add_argument("-s", "--set-seed", action="store_true", help="set seed")

    args = parser.parse_args()

    train_csv_path = args.train_csv_path if not args.debug else "/data/ziz/chho/msc2021/debug/training.csv"
    test_csv_path = args.test_csv_path if not args.debug else "/data/ziz/chho/msc2021/debug/testing.csv"

    assert args.repeats >= args.repeats_start_from, "--repeats must be >= --repeats-start-from"
    # Experiment
    for rep in range(args.repeats_start_from, args.repeats+1):

        if args.set_seed:
            reset_random_seeds(rep)
            print("seed is set as {}".format(rep))

        save_pred = args.save_pred if rep == args.repeats_start_from else False
        save_data = args.save_data if rep == args.repeats_start_from else False

        for exp_idx in args.experiments:

            if args.set_seed:
                reset_random_seeds(rep)
                print("seed is set as {}".format(rep))
            print()
            print("Repeat - ", rep, "Experiment -", exp_idx)
            print()

            # Initialise
            exp_name = "Experiment_" + str(exp_idx)
            exp_dict = getattr(experiments, exp_name)()

            config_path = os.path.join(args.config_dir, exp_dict.get("config_name"))
            config_path = config_path if config_path.split(".")[-1] == "yaml" else config_path + ".yaml"
            config = parse_config(config_path)

            if args.cuda:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args.cuda])
                config["GPU"] = args.cuda

            if args.data_path:
                config["Data"][args.data_path]["save_path"] = args.data_path

            # Restore checkpoint for training
            if args.ckpt_train_save_dir:
                config["Train"]["save"]["ckpt_save_dir"] = args.ckpt_train_save_dir
            if args.ckpt_train_restore_path:
                config["Train"]["save"]["ckpt_restore_path"] = args.ckpt_train_restore_path
            config["Train"]["save"]["restore_from_ckpt"] = args.restore_from_checkpoint | config["Train"]["save"]["restore_from_ckpt"]

            # Data loading
            exp_dict_data = exp_dict.get("data")
            Load_fn = exp_dict_data.get("load_fn")
            data = Load_fn(config=config, **exp_dict_data)
            train_data = data.get("train_data")
            val_data = data.get("val_data")
            test_data = data.get("test_data")
            extra_data = data.get("extra_data")

            # Model parsing
            exp_dict_learner = exp_dict.get("learner")
            model = exp_dict_learner.get("model")
            learner = exp_dict_learner.get("learner")
            model_name = exp_dict_learner.get("model_name")
            learner_load_fn = exp_dict_learner.get("load_fn")
            name = exp_name + "_repeat_" + str(rep)
            learner = learner_load_fn(config=config, name=name, **exp_dict_learner)

            # Training
            if not args.no_train:
                start_time = time()

                train_fn = exp_dict.get("train_fn")
                learner = train_fn(
                    learner = learner,
                    train_data = train_data, 
                    val_data = val_data, 
                    test_data = test_data,
                    extra_data = extra_data,
                    train = True)

                end_time = time()

                # remove unuseful checkpoints
                best_epoch = int(learner.best_epoch)
                last_epoch = int(learner.epoch_end)
                ckpt_dir = learner._ckpt_save_dir
                ckpt_prefix = learner._ckpt_save_prefix
                if not args.keep_ckpts:
                    remove_checkpoints(best_epoch=best_epoch, last_epoch=last_epoch, ckpt_dir=ckpt_dir, ckpt_prefix=ckpt_prefix)

            # Testing
            if not args.no_test:
                if args.no_train:
                    train_fn = exp_dict.get("train_fn") #TODO: avoid loading all data
                    learner = train_fn(
                        learner = learner,
                        train_data = train_data, 
                        val_data = val_data, 
                        test_data = test_data,
                        extra_data = extra_data,
                        train = False)


                test_fn = exp_dict.get("test_fn")

                test_args = dict(
                    test_size = args.test_size,
                    checkpoint_path = args.ckpt_test_restore_path,
                    use_exact_ckpt = args.use_exact_ckpt,
                    result_save_dir = args.save_dir,
                    result_save_filename = args.save_filename,
                    save_pred = save_pred,
                    save_data = save_data
                )
                learner_test, test_output = test_fn(learner=learner, **test_args)
                restore_path, result_save_path, output_mean_res = test_output


            # Save results
            base_dict = dict(
                exp_name = exp_name,
                repeat = rep,
                config_dir = args.config_dir,
                data_source = args.data_source,
                comment = args.comment,
                exp_info = exp_dict.get("other").get("info"),
            )

            config_df = pd.json_normalize(config, sep="_")
            base_dict.update(config_df.to_dict(orient="records")[0])
            

            if not args.no_train:
                train_dict = base_dict.copy()
                train_dict.update(dict(
                    duration = end_time - start_time,
                    start_datetime = learner.current_time,
                    epoch_start = learner.epoch_start,
                    epoch_end = learner.epoch_end,
                    ckpt_restore_path_exact = learner._ckpt_restore_path,
                    ckpt_save_dir_exact = learner._ckpt_save_dir,
                    best_epoch = int(learner.best_epoch),
                    has_test = not args.no_test,
                ))


            if not args.no_test:
                test_dict = base_dict.copy()
                test_dict.update(dict(
                    test_datetime = learner_test.test_time,
                    train_datetime = os.path.basename(os.path.dirname(restore_path)),
                    restore_path = restore_path,
                    result_save_path = result_save_path,
                    test_size = args.test_size,
                    save_pred = save_pred,
                    save_data = save_data,
                    has_train = not args.no_train,
                ))

                test_dict.update(output_mean_res) # save result
                
            # Save as csv
            if not args.no_train:
                if os.path.isfile(train_csv_path):
                    df = pd.read_csv(train_csv_path)
                else:
                    df = pd.DataFrame()

                train_frame = pd.Series(train_dict).to_frame().transpose()
                df_train = pd.concat([df, train_frame])

                df_train.to_csv(train_csv_path, index=False)
            
            if not args.no_test:
                if os.path.isfile(test_csv_path):
                    df = pd.read_csv(test_csv_path)
                else:
                    df = pd.DataFrame()

                test_frame = pd.Series(test_dict).to_frame().transpose()
                df_test = pd.concat([df, test_frame])

                df_test.to_csv(test_csv_path, index=False)

            if rep == 1:
                # Copy python files

                if not args.no_train:
                    code_save_dir = learner._ckpt_save_dir
                else:
                    code_save_dir = os.path.dirname(result_save_path)

                if args.no_train and not args.no_test:
                    code_save_dir = os.path.join(code_save_dir, "code"+learner_test.test_time)
                else:
                    code_save_dir = os.path.join(code_save_dir, "code")

                if not os.path.isdir(code_save_dir):
                    os.mkdir(code_save_dir)

                file_dirname = os.path.dirname(os.path.realpath(__file__))
                copytree(file_dirname, code_save_dir)







