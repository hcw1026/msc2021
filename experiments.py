from data.gp_regression import DataProvider as gp_provider
from data.leo_imagenet import DataProvider as imagenet_provider
from learner import ImageNetLearner, GPLearner
from model import MetaFunClassifier, MetaFunRegressor, MetaFunClassifierV2, MetaFunRegressorV2, MetaFunRegressorV3
from sklearn.gaussian_process import kernels
from run import GPTrain, GPTest, GPLearnerLoad, GPDataLoad, ImageNetTrain, ImageNetTest, ImageNetLearnerLoad, ImageNetDataLoad, GPDataLoadTE

############################################################################################################################
# Regression
############################################################################################################################

#### Experiment 1 ##########################################################################################################

def Experiment_1a():

    return dict(
        config_name = "config1",
        learner = dict(
            learner = GPLearner,
            model = MetaFunRegressor,
            load_fn = GPLearnerLoad,
            model_name = "MetaFunRegressor",
        ),
        train_fn = GPTrain,
        test_fn = GPTest,
        data = dict( # for data loading function parser in run.py
            load_fn = GPDataLoad,
            dataprovider = gp_provider,
            load_type = "single",
            custom_kernels = {"RBF_Kernel":kernels.RBF(length_scale=(0.2))}, 
            custom_kernels_merge = False, 
        ),
        other = dict( # for saving
            info = "Simple MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, parametric init, deep-se kernel",
        )
        )

def Experiment_1b():

    output_dict = Experiment_1a()
    output_dict["data"]["custom_kernels"] = {"Periodic_Kernel":kernels.ExpSineSquared(length_scale=0.5, periodicity=0.5)}
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, parametric init, deep-se kernel"
    return output_dict

def Experiment_1bii():

    output_dict = Experiment_1b()
    output_dict["config_name"] = "config1b"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, parametric init, deep-se kernel, with lower lr than Experiment 1b"
    return output_dict

def Experiment_1biii():

    output_dict = Experiment_1b()
    output_dict["config_name"] = "config1bii"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, parametric init, deep-se kernel, with 0.1 lengthscale initialisation instead"
    return output_dict

def Experiment_1c():

    output_dict = Experiment_1a()
    output_dict["data"]["custom_kernels"] = {"Noisy_Matern_Kernel":kernels.WhiteKernel(noise_level=0.1) + kernels.Matern(length_scale=0.2, nu=1.5)}
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Noisy Matern Kernel, with decoder, same neural iteration, parametric init, deep-se kernel"
    return output_dict


def Experiment_1d():

    output_dict = Experiment_1a()
    output_dict["data"]["custom_kernels"] = {"Variable_Matern_Kernel":kernels.Matern(length_scale_bounds=(0.01, 0.3), nu=1.5)}
    output_dict["data"]["load_type"] = "var_hyp"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, parametric init, deep-se kernel"
    return output_dict

def Experiment_1e():

    output_dict = Experiment_1a()
    output_dict["data"]["custom_kernels"] = {"RBF_Kernel":kernels.RBF(length_scale=(0.2)), "Periodic_Kernel":kernels.ExpSineSquared(length_scale=0.5, periodicity=0.5), "Noisy_Matern_Kernel":kernels.WhiteKernel(noise_level=0.1) + kernels.Matern(length_scale=0.2, nu=1.5)}
    output_dict["data"]["load_type"] = "var_kernel"
    output_dict["data"]["custom_kernels_merge"] = True
    output_dict["other"]["info"] = "Simple MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, same neural iteration, parametric init, deep-se kernel"
    return output_dict



#### Experiment 2 ##########################################################################################################

def Experiment_2a():
    output_dict = Experiment_1a()
    output_dict["config_name"] = "config2"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, parametric init, attention, logprob"
    return output_dict

def Experiment_2b():
    output_dict = Experiment_1b()
    output_dict["config_name"] = "config2"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, parametric init, attention, logprob"
    return output_dict

def Experiment_2c():
    output_dict = Experiment_1c()
    output_dict["config_name"] = "config2"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Noisy Matern Kernel, with decoder, same neural iteration, parametric init, attention, logprob"
    return output_dict

def Experiment_2d():
    output_dict = Experiment_1d()
    output_dict["config_name"] = "config2"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, parametric init, attention, logprob"
    return output_dict

def Experiment_2e():
    output_dict = Experiment_1e()
    output_dict["config_name"] = "config2"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, same neural iteration, parametric init, deep-se kernel, logprob"
    return output_dict



#### Experiment 3 ##########################################################################################################

def Experiment_3a():
    output_dict = Experiment_1a()
    output_dict["config_name"] = "config3"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob"
    return output_dict

def Experiment_3b():
    output_dict = Experiment_1b()
    output_dict["config_name"] = "config3"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob"
    return output_dict

def Experiment_3c():
    output_dict = Experiment_1c()
    output_dict["config_name"] = "config3"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob"
    return output_dict

def Experiment_3d():
    output_dict = Experiment_1d()
    output_dict["config_name"] = "config3"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob"
    return output_dict

def Experiment_3e():
    output_dict = Experiment_1e()
    output_dict["config_name"] = "config3"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob"
    return output_dict



#### Experiment 4 ##########################################################################################################

def Experiment_4a():
    output_dict = Experiment_3a()
    output_dict["config_name"] = "config4"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, parametric init, attention, logprob"
    return output_dict

def Experiment_4b():
    output_dict = Experiment_3b()
    output_dict["config_name"] = "config4"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, parametric init, attention, logprob"
    return output_dict

def Experiment_4c():
    output_dict = Experiment_3c()
    output_dict["config_name"] = "config4"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, parametric init, attention, logprob"
    return output_dict

def Experiment_4d():
    output_dict = Experiment_3d()
    output_dict["config_name"] = "config4"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, parametric init, attention, logprob"
    return output_dict

def Experiment_4e():
    output_dict = Experiment_3e()
    output_dict["config_name"] = "config4"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob"
    return output_dict



#### Experiment 5 ##########################################################################################################

def Experiment_5a():
    output_dict = Experiment_1a()
    output_dict["config_name"] = "config5"
    output_dict["learner"] = dict(
        learner = GPLearner,
        model = MetaFunRegressorV2,
        load_fn = GPLearnerLoad,
        model_name = "MetaFunRegressorV2",
        )
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_5b():
    output_dict = Experiment_1b()
    output_dict["config_name"] = "config5"
    output_dict["learner"] = dict(
        learner = GPLearner,
        model = MetaFunRegressorV2,
        load_fn = GPLearnerLoad,
        model_name = "MetaFunRegressorV2",
        )
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_5c():
    output_dict = Experiment_1c()
    output_dict["config_name"] = "config5"
    output_dict["learner"] = dict(
        learner = GPLearner,
        model = MetaFunRegressorV2,
        load_fn = GPLearnerLoad,
        model_name = "MetaFunRegressorV2",
        )
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_5d():
    output_dict = Experiment_1d()
    output_dict["config_name"] = "config5"
    output_dict["learner"] = dict(
        learner = GPLearner,
        model = MetaFunRegressorV2,
        load_fn = GPLearnerLoad,
        model_name = "MetaFunRegressorV2",
        )
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_5e():
    output_dict = Experiment_1e()
    output_dict["config_name"] = "config5"
    output_dict["learner"] = dict(
        learner = GPLearner,
        model = MetaFunRegressorV2,
        load_fn = GPLearnerLoad,
        model_name = "MetaFunRegressorV2",
        )
    output_dict["other"]["info"] = "Simple MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable"
    return output_dict



#### Experiment 6 ##########################################################################################################

def Experiment_6a():
    output_dict = Experiment_5a()
    output_dict["config_name"] = "config6"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, parametric init, attention, logprob, with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_6b():
    output_dict = Experiment_5b()
    output_dict["config_name"] = "config6"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, parametric init, attention, logprob, with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_6c():
    output_dict = Experiment_5c()
    output_dict["config_name"] = "config6"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, parametric init, attention, logprob, with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_6d():
    output_dict = Experiment_5d()
    output_dict["config_name"] = "config6"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, parametric init, attention, logprob, with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_6e():
    output_dict = Experiment_5e()
    output_dict["config_name"] = "config6"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable"
    return output_dict



#### Experiment 7 ##########################################################################################################

def Experiment_7a():
    output_dict = Experiment_5a()
    output_dict["config_name"] = "config7"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, reuse_across_epochs"
    return output_dict

def Experiment_7b():
    output_dict = Experiment_5b()
    output_dict["config_name"] = "config7"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, reuse_across_epochs"
    return output_dict

def Experiment_7c():
    output_dict = Experiment_5c()
    output_dict["config_name"] = "config7"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, reuse_across_epochs"
    return output_dict

def Experiment_7d():
    output_dict = Experiment_5d()
    output_dict["config_name"] = "config7"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, reuse_across_epochs"
    return output_dict

def Experiment_7e():
    output_dict = Experiment_5e()
    output_dict["config_name"] = "config7"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, reuse_across_epochs"
    return output_dict



#### Experiment 8 ##########################################################################################################

def Experiment_8a():
    output_dict = Experiment_5a()
    output_dict["config_name"] = "config8"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with stddev scaling 0.01"
    return output_dict

def Experiment_8b():
    output_dict = Experiment_5b()
    output_dict["config_name"] = "config8"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with stddev scaling 0.01"
    return output_dict

def Experiment_8c():
    output_dict = Experiment_5c()
    output_dict["config_name"] = "config8"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with stddev scaling 0.01"
    return output_dict

def Experiment_8d():
    output_dict = Experiment_5d()
    output_dict["config_name"] = "config8"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with stddev scaling 0.01"
    return output_dict

def Experiment_8e():
    output_dict = Experiment_5e()
    output_dict["config_name"] = "config8"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with stddev scaling 0.01"
    return output_dict


#### Experiment 9 ##########################################################################################################

def Experiment_9a():
    output_dict = Experiment_5a()
    output_dict["config_name"] = "config9"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with trainable fourier features of size 10 and stddev 10"
    return output_dict

def Experiment_9aii():
    output_dict = Experiment_5b()
    output_dict["config_name"] = "config9"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with trainable fourier features of size 10 and stddev 10"
    return output_dict

def Experiment_9aiii():
    output_dict = Experiment_5d()
    output_dict["config_name"] = "config9"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with trainable fourier features of size 10 and stddev 10"
    return output_dict

def Experiment_9b():
    output_dict = Experiment_5a()
    output_dict["config_name"] = "config9b"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 1 non-trainable"
    return output_dict

def Experiment_9bii():
    output_dict = Experiment_5b()
    output_dict["config_name"] = "config9b"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 1 non-trainable"
    return output_dict

def Experiment_9biii():
    output_dict = Experiment_5d()
    output_dict["config_name"] = "config9b"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 1 non-trainable"
    return output_dict

def Experiment_9c():
    output_dict = Experiment_5a()
    output_dict["config_name"] = "config9c"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 100 non-trainable"
    return output_dict

def Experiment_9cii():
    output_dict = Experiment_5b()
    output_dict["config_name"] = "config9c"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 100 non-trainable"
    return output_dict

def Experiment_9ciii():
    output_dict = Experiment_5d()
    output_dict["config_name"] = "config9c"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 100 non-trainable"
    return output_dict

def Experiment_9d():
    output_dict = Experiment_5a()
    output_dict["config_name"] = "config9d"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 5 and stddev 10 non-trainable"
    return output_dict

def Experiment_9dii():
    output_dict = Experiment_5b()
    output_dict["config_name"] = "config9d"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 5 and stddev 10 non-trainable"
    return output_dict

def Experiment_9diii():
    output_dict = Experiment_5d()
    output_dict["config_name"] = "config9d"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 5 and stddev 10 non-trainable"
    return output_dict

def Experiment_9e():
    output_dict = Experiment_5a()
    output_dict["config_name"] = "config9e"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 50 and stddev 10 non-trainable"
    return output_dict

def Experiment_9eii():
    output_dict = Experiment_5b()
    output_dict["config_name"] = "config9e"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 50 and stddev 10 non-trainable"
    return output_dict

def Experiment_9eiii():
    output_dict = Experiment_5d()
    output_dict["config_name"] = "config9e"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 50 and stddev 10 non-trainable"
    return output_dict



#### Experiment 10 ##########################################################################################################

def Experiment_10a():
    output_dict = Experiment_8a()
    output_dict["config_name"] = "config10"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with 100 epochs without early stopping"
    return output_dict

def Experiment_10b():
    output_dict = Experiment_8b()
    output_dict["config_name"] = "config10"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with 100 epochs without early stopping"
    return output_dict

def Experiment_10c():
    output_dict = Experiment_8c()
    output_dict["config_name"] = "config10"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with 100 epochs without early stopping"
    return output_dict

def Experiment_10d():
    output_dict = Experiment_8d()
    output_dict["config_name"] = "config10"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with 100 epochs without early stopping"
    return output_dict

def Experiment_10e():
    output_dict = Experiment_8e()
    output_dict["config_name"] = "config10"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with 100 epochs without early stopping"
    return output_dict



#### Experiment 11 ##########################################################################################################

def Experiment_11a():
    output_dict = Experiment_8a()
    output_dict["learner"]["model"] = MetaFunRegressorV3
    output_dict["Learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["data"]["load_fn"] = GPDataLoadTE
    output_dict["data"]["offsets"] = [0.1]
    output_dict["config_name"] = "config11"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_11b():
    output_dict = Experiment_8b()
    output_dict["learner"]["model"] = MetaFunRegressorV3
    output_dict["Learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["data"]["load_fn"] = GPDataLoadTE
    output_dict["data"]["offsets"] = [0.1]
    output_dict["config_name"] = "config11"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero,  with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_11c():
    output_dict = Experiment_8c()
    output_dict["learner"]["model"] = MetaFunRegressorV3
    output_dict["Learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["data"]["load_fn"] = GPDataLoadTE
    output_dict["data"]["offsets"] = [0.1]
    output_dict["config_name"] = "config11"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero,  with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_11d():
    output_dict = Experiment_8d()
    output_dict["learner"]["model"] = MetaFunRegressorV3
    output_dict["Learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["data"]["load_fn"] = GPDataLoadTE
    output_dict["data"]["offsets"] = [0.1]
    output_dict["config_name"] = "config11"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero,  with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_11e():
    output_dict = Experiment_8e()
    output_dict["learner"]["model"] = MetaFunRegressorV3
    output_dict["Learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["data"]["load_fn"] = GPDataLoadTE
    output_dict["data"]["offsets"] = [0.1]
    output_dict["config_name"] = "config11"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero,  with fourier features size 10 and stddev 10 non-trainable"
    return output_dict



############################################################################################################################
# Classification
############################################################################################################################

#### Experiment cls1 ##########################################################################################################

def Experiment_cls1a(): #imagenet experiments

    return dict(
        config_name = "config_cls1",
        learner = dict(
            learner = ImageNetLearner,
            model = MetaFunClassifier,
            data_source="leo_imagenet",
            load_fn = ImageNetLearnerLoad,
            model_name = "MetaFunClassifier",
        ),
        train_fn = ImageNetTrain,
        test_fn = ImageNetTest,
        data = dict( # for data loading function parser in run.py
            load_fn = ImageNetDataLoad,
            dataprovider = imagenet_provider,
        ),
        other = dict( # for saving
            info = "MetaFunClassifier on tieredimagenet with deep se kernel and neural update 5-way-1-shot",
        )
        )

def Experiment_cls1b(): #imagenet experiments

    output_dict = Experiment_cls1a()
    output_dict["config_name"] = "config_cls1b"
    output_dict["other"]["info"] = "MetaFunClassifier on tieredimagenet with attention and neural update 5-way-1-shot"
    return output_dict

def Experiment_cls1c(): #imagenet experiments

    output_dict = Experiment_cls1a()
    output_dict["config_name"] = "config_cls1c"
    output_dict["other"]["info"] = "MetaFunClassifier on tieredimagenet with kernel and neural update 5-way-5-shot"
    return output_dict

def Experiment_cls1d(): #imagenet experiments

    output_dict = Experiment_cls1a()
    output_dict["config_name"] = "config_cls1d"
    output_dict["other"]["info"] = "MetaFunClassifier on tieredimagenet with attention and neural update 5-way-5-shot"
    return output_dict

############################################################################################################################
# Debug
############################################################################################################################

def Experiment_0(): #debug

    return dict(
        config_name = "debug_copy",
        learner = dict(
            learner = GPLearner,
            model = MetaFunRegressor,
            load_fn = GPLearnerLoad,
            model_name = "MetaFunRegressor",
        ),
        train_fn = GPTrain,
        test_fn = GPTest,
        data = dict( # for data loading function parser in run.py
            load_fn = GPDataLoad,
            dataprovider = gp_provider,
            load_type = "single",
            custom_kernels = {"RBF_Kernel":kernels.RBF(length_scale=(0.2))}, 
            custom_kernels_merge = False, 
        ),
        other = dict( # for saving
            info = "Simple MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, parametric init, deep-se kernel",
        )
        )

def Experiment_cls0(): #imagenet experiments

    return dict(
        config_name = "debug_copy",
        learner = dict(
            learner = ImageNetLearner,
            model = MetaFunClassifier,
            data_source="leo_imagenet",
            load_fn = ImageNetLearnerLoad,
            model_name = "MetaFunClassifier",
        ),
        train_fn = ImageNetTrain,
        test_fn = ImageNetTest,
        data = dict( # for data loading function parser in run.py
            load_fn = ImageNetDataLoad,
            dataprovider = imagenet_provider,
        ),
        other = dict( # for saving
            info = "MetaFunClassifier on tieredimagenet with deep se kernel and neural update",
        )
        )