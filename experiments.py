from data.gp_regression import DataProvider as gp_provider
from data.leo_imagenet import DataProvider as imagenet_provider
from learner import ImageNetLearner, GPLearner
from model import MetaFunClassifier, MetaFunRegressor, MetaFunClassifierV2, MetaFunRegressorV2, MetaFunRegressorV3, MetaFunRegressorV3b, MetaFunRegressorGLV3, MetaFunRegressorV4, MetaFunRegressorGLV4
from sklearn.gaussian_process import kernels
from run import GPTrain, GPTest, GPLearnerLoad, GPDataLoad, ImageNetTrain, ImageNetTest, ImageNetLearnerLoad, ImageNetDataLoad, GPDataLoadTE

############################################################################################################################
# Regression
############################################################################################################################

#### Experiment 1 ##########################################################################################################
#original model with deep-se

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
#original model with attention

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
#original model + independent update + deep-se

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
#original model + independent update + attention

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
#original model + independent update + ff + deep-se

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
#original model + independent update + ff + attention

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
#original model + independent update + ff + deep-se + reuse data across epochs

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
#original model + independent update + ff + deep-se + lower stddev initial constant to 0.01

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
#original model + independent update + ff + deep-se + testing different ff parameters

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
#original model + independent update + ff + deep-se + lower stddev initial constant to 0.01 + concat x in neural-local updater
#from now on this is Model-A

def Experiment_10a():
    output_dict = Experiment_5a()
    output_dict["config_name"] = "config10"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with stddev scaling 0.01, and concatenate x to neural-local-updater"
    return output_dict

def Experiment_10b():
    output_dict = Experiment_5b()
    output_dict["config_name"] = "config10"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with stddev scaling 0.01, and concatenate x to neural-local-updater"
    return output_dict

def Experiment_10c():
    output_dict = Experiment_5c()
    output_dict["config_name"] = "config10"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with stddev scaling 0.01, and concatenate x to neural-local-updater"
    return output_dict

def Experiment_10d():
    output_dict = Experiment_5d()
    output_dict["config_name"] = "config10"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with stddev scaling 0.01, and concatenate x to neural-local-updater"
    return output_dict

def Experiment_10e():
    output_dict = Experiment_5e()
    output_dict["config_name"] = "config10"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with stddev scaling 0.01, and concatenate x to neural-local-updater"
    return output_dict



#### Experiment 11 ##########################################################################################################
#Simple Model + ff + se kernel + no x concatenation in neural updater + constant intialiser + repr_as_input for prediction

def Experiment_11a():
    output_dict = Experiment_8a()
    output_dict["learner"]["model"] = MetaFunRegressorV3
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["data"]["load_fn"] = GPDataLoadTE
    output_dict["data"]["offsets"] = [0.1]
    output_dict["config_name"] = "config11"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_11aii():
    output_dict = Experiment_11a()
    output_dict["config_name"] = "config11b"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_11b():
    output_dict = Experiment_8b()
    output_dict["learner"]["model"] = MetaFunRegressorV3
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["data"]["load_fn"] = GPDataLoadTE
    output_dict["data"]["offsets"] = [0.1]
    output_dict["config_name"] = "config11"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero,  with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_11bii():
    output_dict = Experiment_11b()
    output_dict["config_name"] = "config11b"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_11c():
    output_dict = Experiment_8c()
    output_dict["learner"]["model"] = MetaFunRegressorV3
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["data"]["load_fn"] = GPDataLoadTE
    output_dict["data"]["offsets"] = [0.1]
    output_dict["config_name"] = "config11"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero,  with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_11cii():
    output_dict = Experiment_11c()
    output_dict["config_name"] = "config11b"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_11d():
    output_dict = Experiment_8d()
    output_dict["learner"]["model"] = MetaFunRegressorV3
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["data"]["load_fn"] = GPDataLoadTE
    output_dict["data"]["offsets"] = [0.1]
    output_dict["config_name"] = "config11"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero,  with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_11dii():
    output_dict = Experiment_11d()
    output_dict["config_name"] = "config11b"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_11e():
    output_dict = Experiment_8e()
    output_dict["learner"]["model"] = MetaFunRegressorV3
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["data"]["load_fn"] = GPDataLoadTE
    output_dict["data"]["offsets"] = [0.1]
    output_dict["config_name"] = "config11"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero,  with fourier features size 10 and stddev 10 non-trainable"
    return output_dict

def Experiment_11eii():
    output_dict = Experiment_11e()
    output_dict["config_name"] = "config11b"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict



#### Experiment 12 ##########################################################################################################
#Model-A without early stopping

def Experiment_12a():
    output_dict = Experiment_8a()
    output_dict["config_name"] = "config12"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with 100 epochs without early stopping"
    return output_dict

def Experiment_12b():
    output_dict = Experiment_8b()
    output_dict["config_name"] = "config12"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with 100 epochs without early stopping"
    return output_dict

def Experiment_12c():
    output_dict = Experiment_8c()
    output_dict["config_name"] = "config12"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with 100 epochs without early stopping"
    return output_dict

def Experiment_12d():
    output_dict = Experiment_8d()
    output_dict["config_name"] = "config12"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with 100 epochs without early stopping"
    return output_dict

def Experiment_12e():
    output_dict = Experiment_8e()
    output_dict["config_name"] = "config12"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with 100 epochs without early stopping"
    return output_dict



#### Experiment 13 ##########################################################################################################
#Translation equivariant trials

def Experiment_13a():
    output_dict = Experiment_11a()
    output_dict["config_name"] = "config13"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, constant init, rff kernel with None mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13aii():
    output_dict = Experiment_11a()
    output_dict["config_name"] = "config13b"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, constant init, rff kernel with deepset1 mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13aiii():
    output_dict = Experiment_11a()
    output_dict["config_name"] = "config13c"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, constant init, rff kernel with sab mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13aiv():
    output_dict = Experiment_11a()
    output_dict["config_name"] = "config13d"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, constant init, deepse kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13av():
    output_dict = Experiment_11a()
    output_dict["config_name"] = "config13e"
    output_dict["learner"]["model"] = MetaFunRegressorV3b
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3b"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with RBF Kernel, with decoder, independent neural iteration, parametric init, rff kernel, logprob, nueral local updater with x appending, normal prediction, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13b():
    output_dict = Experiment_11b()
    output_dict["config_name"] = "config13"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, constant init, rff kernel with None mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13bii():
    output_dict = Experiment_11b()
    output_dict["config_name"] = "config13b"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, constant init, rff kernel with deepset1 mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13biii():
    output_dict = Experiment_11b()
    output_dict["config_name"] = "config13c"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, constant init, rff kernel with sab mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13biv():
    output_dict = Experiment_11b()
    output_dict["config_name"] = "config13d"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, constant init, deepse kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13bv():
    output_dict = Experiment_11b()
    output_dict["config_name"] = "config13e"
    output_dict["learner"]["model"] = MetaFunRegressorV3b
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3b"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Periodic Kernel, with decoder, independent neural iteration, parametric init, rff kernel, logprob, nueral local updater with x appending, normal prediction, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13c():
    output_dict = Experiment_11c()
    output_dict["config_name"] = "config13"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, constant init, rff kernel with None mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13cii():
    output_dict = Experiment_11c()
    output_dict["config_name"] = "config13b"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, constant init, rff kernel with deepset1 mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13ciii():
    output_dict = Experiment_11c()
    output_dict["config_name"] = "config13c"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, constant init, rff kernel with sab mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13civ():
    output_dict = Experiment_11c()
    output_dict["config_name"] = "config13d"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, constant init, deepse kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13cv():
    output_dict = Experiment_11c()
    output_dict["config_name"] = "config13e"
    output_dict["learner"]["model"] = MetaFunRegressorV3b
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3b"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Noisy Matern Kernel, with decoder, independent neural iteration, parametric init, rff kernel, logprob, nueral local updater with x appending, normal prediction, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13d():
    output_dict = Experiment_11d()
    output_dict["config_name"] = "config13"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, constant init, rff kernel with None mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13dii():
    output_dict = Experiment_11d()
    output_dict["config_name"] = "config13b"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, constant init, rff kernel with deepset1 mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13diii():
    output_dict = Experiment_11d()
    output_dict["config_name"] = "config13c"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, constant init, rff kernel with sab mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13div():
    output_dict = Experiment_11d()
    output_dict["config_name"] = "config13d"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, constant init, deepse kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13dv():
    output_dict = Experiment_11d()
    output_dict["config_name"] = "config13e"
    output_dict["learner"]["model"] = MetaFunRegressorV3b
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3b"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with Variable Matern Kernel, with decoder, independent neural iteration, parametric init, rff kernel, logprob, nueral local updater with x appending, normal prediction, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13e():
    output_dict = Experiment_11e()
    output_dict["config_name"] = "config13"

    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, constant init, rff kernel with None mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13eii():
    output_dict = Experiment_11e()
    output_dict["config_name"] = "config13b"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, constant init, rff kernel with deepset1 mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13eiii():
    output_dict = Experiment_11e()
    output_dict["config_name"] = "config13c"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, constant init, rff kernel with sab mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13eiv():
    output_dict = Experiment_11e()
    output_dict["config_name"] = "config13d"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, constant init, deepse kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict

def Experiment_13ev():
    output_dict = Experiment_11e()
    output_dict["config_name"] = "config13e"
    output_dict["learner"]["model"] = MetaFunRegressorV3b
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3b"
    output_dict["other"]["info"] = "Simple translation equivariant experiment - MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, independent neural iteration, parametric init,  kernel, logprob, nueral local updater with x appending, normal prediction, with fourier features size 10 and stddev 10 non-trainable, without repr_as_input"
    return output_dict



#### Experiment 14 ##########################################################################################################
#Translatin equivariant

def Experiment_14a():
    output_dict = Experiment_11a()
    output_dict["data"]["offsets"] = [100]
    output_dict["config_name"] = "config14"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with None mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14aii():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14b"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14aiii():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14c"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with SAB mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14aiv():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14d"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, deepse kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14av():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14e"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14as():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14eii"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, se kernel with correct initial lengthscale, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14as2():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14eiii"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, se kernel with correct initial lengthscale that is untrainable, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14avi():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14f"
    output_dict["learner"]["model"] = MetaFunRegressorV3b
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3b"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, parametric init, deepse kernel, logprob, nueral local updater with x appending, leo prediction, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14avii():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14g"
    output_dict["learner"]["model"] = MetaFunRegressorV3b
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3b"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, parametric init, sab kernel, logprob, nueral local updater with x appending, leo prediction, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14aviii():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14h"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with none mapping with 50 initial variables, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14aviv():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14i"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with none mapping with 1500 initial variables, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14ax():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14j"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, deepse kernel with 3-embedding layers, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14axi():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14k"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping with nontrainable init, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14axii():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14l"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with SAB mapping with nontrainable init, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14axiii():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14m"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with None mapping (init Uniform(-1,1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14axiv():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14n"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Uniform(-1,1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14axv():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14o"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with SAB mapping (init Uniform(-1,1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14axvi():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14p"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with None mapping (init Normal(0,0.1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14axvii():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14q"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14axviii():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14r"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with SAB mapping (init Normal(0,0.1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14axviv():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14t"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with None mapping (init Normal(0,10)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14axx():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14u"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,10)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14axxi():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config14v"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with SAB mapping (init Normal(0,10)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14b():
    output_dict = Experiment_11b()
    output_dict["data"]["offsets"] = [100]
    output_dict["config_name"] = "config14"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with None mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bii():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14b"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14biii():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14c"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with SAB mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14biv():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14d"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, deepse kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bv():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14e"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bvi():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14f"
    output_dict["learner"]["model"] = MetaFunRegressorV3b
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3b"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, parametric init, deepse kernel, logprob, nueral local updater with x appending, leo prediction, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bvii():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14g"
    output_dict["learner"]["model"] = MetaFunRegressorV3b
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3b"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, parametric init, sab kernel, logprob, nueral local updater with x appending, leo prediction, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bviii():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14h"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with none mapping with 50 initial variables, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bviv():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14i"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with none mapping with 1500 initial variables, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bx():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14j"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, deepse kernel with 3-embedding layers, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bxi():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14k"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping with nontrainable init, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bxii():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14l"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with SAB mapping with nontrainable init, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bxiii():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14m"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with None mapping (init Uniform(-1,1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bxiv():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14n"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Uniform(-1,1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bxv():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14o"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with SAB mapping (init Uniform(-1,1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bxvi():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14p"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with None mapping (init Normal(0,0.1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bxvii():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14q"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bxviii():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14r"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with SAB mapping (init Normal(0,0.1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bxviv():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14t"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with None mapping (init Normal(0,10)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bxx():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14u"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,10)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14bxxi():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config14v"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with SAB mapping (init Normal(0,10)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14c():
    output_dict = Experiment_11d()
    output_dict["data"]["offsets"] = [100]
    output_dict["config_name"] = "config14"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with None mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cii():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14b"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14ciii():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14c"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with SAB mapping, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14civ():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14d"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, deepse kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cv():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14e"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, se kernel, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cvi():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14f"
    output_dict["learner"]["model"] = MetaFunRegressorV3b
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3b"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, parametric init, deepse kernel, logprob, nueral local updater with x appending, leo prediction, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cvii():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14g"
    output_dict["learner"]["model"] = MetaFunRegressorV3b
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3b"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, parametric init, sab kernel, logprob, nueral local updater with x appending, leo prediction, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cviii():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14h"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with none mapping with 50 initial variables, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cviv():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14i"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with none mapping with 1500 initial variables, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cx():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14j"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, deepse kernel with 3-embedding layers, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cxi():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14k"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping with nontrainable init, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cxii():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14l"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with SAB mapping with nontrainable init, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cxiii():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14m"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with None mapping (init Uniform(-1,1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cxiv():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14n"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Uniform(-1,1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cxv():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14o"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with SAB mapping (init Uniform(-1,1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cxvi():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14p"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with None mapping (init Normal(0,0.1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cxvii():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14q"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cxviii():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14r"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with SAB mapping (init Normal(0,0.1)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cxviv():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14t"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with None mapping (init Normal(0,10)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cxx():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14u"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,10)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_14cxxi():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config14v"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with SAB mapping (init Normal(0,10)), logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict


#### Experiment 15 ##########################################################################################################
#Model-A with n_same_samples=20 instead

def Experiment_15a():
    output_dict = Experiment_5a()
    output_dict["config_name"] = "config15"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with RBF Kernel, with n_same_samples=20, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with stddev scaling 0.01, and concatenate x to neural-local-updater"
    return output_dict

def Experiment_15b():
    output_dict = Experiment_5b()
    output_dict["config_name"] = "config15"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Periodic Kernel, with n_same_samples=20, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with stddev scaling 0.01, and concatenate x to neural-local-updater"
    return output_dict

def Experiment_15c():
    output_dict = Experiment_5c()
    output_dict["config_name"] = "config15"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Noisy Matern Kernel, with n_same_samples=20, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with stddev scaling 0.01, and concatenate x to neural-local-updater"
    return output_dict

def Experiment_15d():
    output_dict = Experiment_5d()
    output_dict["config_name"] = "config15"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with Variable Matern Kernel, with n_same_samples=20, with decoder, independent neural iteration, parametric init, deep se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with stddev scaling 0.01, and concatenate x to neural-local-updater"
    return output_dict

def Experiment_15e():
    output_dict = Experiment_5e()
    output_dict["config_name"] = "config15"
    output_dict["other"]["info"] = "Simple MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with n_same_samples=20, with decoder, independent neural iteration, parametric init, deep-se kernel, logprob, with fourier features size 10 and stddev 10 non-trainable, with stddev scaling 0.01, and concatenate x to neural-local-updater"
    return output_dict




#### Experiment 16 ##########################################################################################################
# global latent
def Experiment_16a():
    output_dict = Experiment_1a()
    output_dict["config_name"] = "config16"
    output_dict["learner"]["model"] = MetaFunRegressorGLV3
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)), logprob_VI, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_16aii():
    output_dict = Experiment_1a()
    output_dict["config_name"] = "config16b"
    output_dict["learner"]["model"] = MetaFunRegressorGLV3
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)), logprob_ML, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_16b():
    output_dict = Experiment_1b()
    output_dict["config_name"] = "config16"
    output_dict["learner"]["model"] = MetaFunRegressorGLV3
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)), logprob_VI, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_16bii():
    output_dict = Experiment_1b()
    output_dict["config_name"] = "config16b"
    output_dict["learner"]["model"] = MetaFunRegressorGLV3
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)), logprob_ML, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_16c():
    output_dict = Experiment_1c()
    output_dict["config_name"] = "config16"
    output_dict["learner"]["model"] = MetaFunRegressorGLV3
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["other"]["info"] = "MetaFunRegressor with Noisy Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)), logprob_VI, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_16cii():
    output_dict = Experiment_1c()
    output_dict["config_name"] = "config16b"
    output_dict["learner"]["model"] = MetaFunRegressorGLV3
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["other"]["info"] = "MetaFunRegressor with Noisy Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)), logprob_ML, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_16d():
    output_dict = Experiment_1d()
    output_dict["config_name"] = "config16"
    output_dict["learner"]["model"] = MetaFunRegressorGLV3
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)), logprob_VI, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_16dii():
    output_dict = Experiment_1d()
    output_dict["config_name"] = "config16b"
    output_dict["learner"]["model"] = MetaFunRegressorGLV3
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)), logprob_ML, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_16e():
    output_dict = Experiment_1e()
    output_dict["config_name"] = "config16"
    output_dict["learner"]["model"] = MetaFunRegressorGLV3
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["other"]["info"] = "MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)), logprob_VI, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_16eii():
    output_dict = Experiment_1e()
    output_dict["config_name"] = "config16b"
    output_dict["learner"]["model"] = MetaFunRegressorGLV3
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3"
    output_dict["other"]["info"] = "MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)), logprob_ML, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict



#### Experiment 17 ##########################################################################################################
def Experiment_17a():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config17"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)) and trainable weight, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_17b():
    output_dict = Experiment_14b()
    output_dict["config_name"] = "config17"
    output_dict["other"]["info"] = "MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)) trainable weight, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_17c():
    output_dict = Experiment_14c()
    output_dict["config_name"] = "config17"
    output_dict["other"]["info"] = "MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)) trainable weight, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

#### Experiment 18 ##########################################################################################################
def Experiment_18a():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config18"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, 3 iters, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)) and trainable weight, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_18aii():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config18b"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, 7 iters, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)) and trainable weight, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_18aiii():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config18c"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, 0.03 inner lr, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)) and trainable weight, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_18aiv():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config18d"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, 0,7 inner lr, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)) and trainable weight, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_18av():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config18d"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel, 10 iters, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)) and trainable weight, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict

def Experiment_18avi():
    output_dict = Experiment_14a()
    output_dict["config_name"] = "config18d"
    output_dict["other"]["info"] = "MetaFunRegressor with RBF Kernel,  0.4 inner lr, with decoder, same neural iteration, constant init, rff kernel with deepset mapping (init Normal(0,0.1)) and trainable weight, logprob, nueral local updater without x appending, prediction with x masked as zero, without fourier features, without repr_as_input"
    return output_dict



#### Experiment 19 ##########################################################################################################
def Experiment_19a():
    return dict(
        config_name = "config19",
        learner = dict(
            learner = GPLearner,
            model = MetaFunRegressorV4,
            load_fn = GPLearnerLoad,
            model_name = "MetaFunRegressorV4",
        ),
        train_fn = GPTrain,
        test_fn = GPTest,
        data = dict( # for data loading function parser in run.py
            load_fn = GPDataLoadTE,
            offsets = [100.],
            dataprovider = gp_provider,
            load_type = "single",
            custom_kernels = {"RBF_Kernel2":kernels.RBF(length_scale=(0.25))}, 
            custom_kernels_merge = False, 
        ),
        other = dict( # for saving
            info = "New rff trial with params matched with ConvCNP paper with RBF kernel",
        )
        )

def Experiment_19b():
    output_dict = Experiment_19a()
    output_dict["data"]["custom_kernels"] = {"Weakly_Periodic_Kernel": kernels.ExpSineSquared(length_scale=1., periodicity=0.25) * kernels.RBF(length_scale=0.5)}
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel"
    return output_dict

def Experiment_19c():
    output_dict = Experiment_19a()
    output_dict["data"]["custom_kernels"] = {"Matern_Kernel": kernels.Matern(length_scale=0.25, nu=2.5)}
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel"
    return output_dict

def Experiment_19d():
    output_dict = Experiment_19a()
    output_dict["data"]["load_type"] = "sawtooth"
    output_dict["config_name"] = "config19ii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth"
    return output_dict

def Experiment_19aii():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19b"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (10 stddev rff)"
    return output_dict

def Experiment_19bii():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19b"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (10 stddev rff)"
    return output_dict

def Experiment_19cii():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19b"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (10 stddev rff)"
    return output_dict

def Experiment_19dii():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19bii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (10 stddev rff)"
    return output_dict

def Experiment_19aiii():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19c"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (10 stddev rff), n_samples=1"
    return output_dict

def Experiment_19biii():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19c"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (10 stddev rff), n_samples=1"
    return output_dict

def Experiment_19ciii():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19c"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (10 stddev rff), n_samples=1"
    return output_dict

def Experiment_19diii():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19cii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (10 stddev rff), n_samples=1"
    return output_dict

def Experiment_19aiv():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19d"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (1 stddev rff), n_samples=1"
    return output_dict

def Experiment_19biv():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19d"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (1 stddev rff), n_samples=1"
    return output_dict

def Experiment_19civ():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19d"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (1 stddev rff), n_samples=1"
    return output_dict

def Experiment_19div():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19dii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (1 stddev rff), n_samples=1"
    return output_dict

def Experiment_19av():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19e"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (uniform rff), n_samples=1"
    return output_dict

def Experiment_19bv():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19e"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (uniform rff), n_samples=1"
    return output_dict

def Experiment_19cv():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19e"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (uniform rff), n_samples=1"
    return output_dict

def Experiment_19dv():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19eii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (uniform rff), n_samples=1"
    return output_dict

def Experiment_19avi():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19f"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (10 stddev rff SAB), n_samples=1"
    return output_dict

def Experiment_19bvi():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19f"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (10 stddev rff SAB), n_samples=1"
    return output_dict

def Experiment_19cvi():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19f"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (10 stddev rff SAB), n_samples=1"
    return output_dict

def Experiment_19dvi():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19fii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (10 stddev rff SAB), n_samples=1"
    return output_dict

def Experiment_19avii():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19g"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (1 stddev rff SAB), n_samples=1"
    return output_dict

def Experiment_19bvii():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19g"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (1 stddev rff SAB), n_samples=1"
    return output_dict

def Experiment_19cvii():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19g"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (1 stddev rff SAB), n_samples=1"
    return output_dict

def Experiment_19dvii():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19gii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (1 stddev rff SAB), n_samples=1"
    return output_dict

def Experiment_19aviii():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19h"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (uniform rff), n_samples=1"
    return output_dict

def Experiment_19bviii():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19h"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (uniform rff), n_samples=1"
    return output_dict

def Experiment_19cviii():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19h"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (uniform rff), n_samples=1"
    return output_dict

def Experiment_19dviii():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19hii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (uniform rff), n_samples=1"
    return output_dict

def Experiment_19aviv():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19i"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (uniform rff), n_samples=1, Deepset 500 simple decoder niter 7"
    return output_dict

def Experiment_19bviv():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19i"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (uniform rff), n_samples=1, Deepset 500 simple decoder niter 7"
    return output_dict

def Experiment_19cviv():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19i"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (uniform rff), n_samples=1, Deepset 500 simple decoder niter 7"
    return output_dict

def Experiment_19dviv():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19iii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (uniform rff), n_samples=1, Deepset 500 simple decoder niter 7"
    return output_dict

def Experiment_19ax():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19j"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (uniform rff), n_samples=1, Deepset 300 simple decoder niter 7"
    return output_dict

def Experiment_19bx():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19j"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (uniform rff), n_samples=1, Deepset 300 simple decoder niter 7"
    return output_dict

def Experiment_19cx():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19j"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (uniform rff), n_samples=1, Deepset 300 simple decoder niter 7"
    return output_dict

def Experiment_19dx():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19jii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (uniform rff), n_samples=1, Deepset 300 simple decoder niter 7"
    return output_dict

def Experiment_19axi():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19k"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (uniform rff), n_samples=1, Deepset 100 simple decoder niter 7"
    return output_dict

def Experiment_19bxi():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19k"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (uniform rff), n_samples=1, Deepset 100 simple decoder niter 7"
    return output_dict

def Experiment_19cvxi():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19k"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (uniform rff), n_samples=1, Deepset 100 simple decoder niter 7"
    return output_dict

def Experiment_19dxi():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19kii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (uniform rff), n_samples=1, Deepset 100 simple decoder niter 7"
    return output_dict

def Experiment_19axii():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19l"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (uniform rff), n_samples=1, SAB 500 simple decoder niter 7"
    return output_dict

def Experiment_19bxii():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19l"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (uniform rff), n_samples=1, SAB 500 simple decoder niter 7"
    return output_dict

def Experiment_19cxii():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19l"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (uniform rff), n_samples=1, SAB 500 simple decoder niter 7"
    return output_dict

def Experiment_19dxii():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19lii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (uniform rff), n_samples=1, SAB 500 simple decoder niter 7"
    return output_dict

def Experiment_19axiii():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19m"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (uniform rff), n_samples=1, SAB 300 simple decoder niter 7"
    return output_dict

def Experiment_19bxiii():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19m"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (uniform rff), n_samples=1, SAB 300 simple decoder niter 7"
    return output_dict

def Experiment_19cxiii():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19m"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (uniform rff), n_samples=1, SAB 300 simple decoder niter 7"
    return output_dict

def Experiment_19dxiii():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19mii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (uniform rff), n_samples=1, SAB 300 simple decoder niter 7"
    return output_dict

def Experiment_19axiv():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19n"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (uniform rff), n_samples=1, SAB 100 simple decoder niter 7"
    return output_dict

def Experiment_19bxiv():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19n"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (uniform rff), n_samples=1, SAB 100 simple decoder niter 7"
    return output_dict

def Experiment_19cxiv():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19n"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (uniform rff), n_samples=1, SAB 100 simple decoder niter 7"
    return output_dict

def Experiment_19dxiv():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19nii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (uniform rff), n_samples=1, SAB 100 simple decoder niter 7"
    return output_dict

def Experiment_19axv():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19o"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (uniform rff), n_samples=1, SAB 1000 simple decoder niter 7"
    return output_dict

def Experiment_19bxv():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19o"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (uniform rff), n_samples=1, SAB 1000 simple decoder niter 7"
    return output_dict

def Experiment_19cxv():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19o"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (uniform rff), n_samples=1, SAB 1000 simple decoder niter 7"
    return output_dict

def Experiment_19dxv():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19oii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (uniform rff), n_samples=1, SAB 1000 simple decoder niter 7"
    return output_dict

def Experiment_19axvi():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19p"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (uniform rff), n_samples=1, Deepset 1000 simple decoder niter 7"
    return output_dict

def Experiment_19bxvi():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19p"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (uniform rff), n_samples=1, Deepset 1000 simple decoder niter 7"
    return output_dict

def Experiment_19cxvi():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19p"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (uniform rff), n_samples=1, Deepset 1000 simple decoder niter 7"
    return output_dict

def Experiment_19dxvi():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19pii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (uniform rff), n_samples=1, Deepset 1000 simple decoder niter 7"
    return output_dict

def Experiment_19axvii():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config19q"
    output_dict["learner"]["model"] = MetaFunRegressorV3b
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3b"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (uniform rff), n_samples=1, deep-se simple decoder niter 7"
    return output_dict

def Experiment_19bxvii():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config19q"
    output_dict["learner"]["model"] = MetaFunRegressorV3b
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3b"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (uniform rff), n_samples=1, deep-se simple decoder niter 7"
    return output_dict

def Experiment_19cxvii():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config19q"
    output_dict["learner"]["model"] = MetaFunRegressorV3b
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3b"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (uniform rff), n_samples=1, deep-se simple decoder niter 7"
    return output_dict

def Experiment_19dxvii():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config19qii"
    output_dict["learner"]["model"] = MetaFunRegressorV3b
    output_dict["learner"]["model_name"] = "MetaFunRegressorV3b"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (uniform rff), n_samples=1, deep-se simple decoder niter 7"
    return output_dict


#### Experiment 20 ##########################################################################################################

def Experiment_20a():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config20"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (uniform rff), n_samples=1, Deepset 300 simple decoder niter 3"
    return output_dict

def Experiment_20b():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config20"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (uniform rff), n_samples=1, Deepset 300 simple decoder niter 3"
    return output_dict

def Experiment_20c():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config20"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (uniform rff), n_samples=1, Deepset 300 simple decoder niter 3"
    return output_dict

def Experiment_20d():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config20ii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (uniform rff), n_samples=1, Deepset 300 simple decoder niter 3"
    return output_dict

def Experiment_20aii():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config20b"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (uniform rff), n_samples=1, Deepset 300 simple decoder niter 5"
    return output_dict

def Experiment_20bii():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config20b"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (uniform rff), n_samples=1, Deepset 300 simple decoder niter 5"
    return output_dict

def Experiment_20cii():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config20b"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (uniform rff), n_samples=1, Deepset 300 simple decoder niter 5"
    return output_dict

def Experiment_20dii():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config20bii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (uniform rff), n_samples=1, Deepset 300 simple decoder niter 5"
    return output_dict

def Experiment_20aiii():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config20c"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (uniform rff), n_samples=1, Deepset 300 simple decoder niter 7 innerlr 0.5"
    return output_dict

def Experiment_20biii():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config20c"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (uniform rff), n_samples=1, Deepset 300 simple decoder niter 7 innerlr 0.5"
    return output_dict

def Experiment_20ciii():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config20c"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (uniform rff), n_samples=1, Deepset 300 simple decoder niter 7 innerlr 0.5"
    return output_dict

def Experiment_20diii():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config20cii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (uniform rff), n_samples=1, Deepset 300 simple decoder niter 7 innerlr 0.5"
    return output_dict

def Experiment_20aiv():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config20d"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (normal rff), n_samples=1, Deepset 300 simple decoder niter 7 innerlr 0.03"
    return output_dict

def Experiment_20biv():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config20d"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (normal rff), n_samples=1, Deepset 300 simple decoder niter 7 innerlr 0.03"
    return output_dict

def Experiment_20civ():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config20d"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (normal rff), n_samples=1, Deepset 300 simple decoder niter 7 innerlr 0.03"
    return output_dict

def Experiment_20div():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config20dii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (normal rff), n_samples=1, Deepset 300 simple decoder niter 7 innerlr 0.03"
    return output_dict

def Experiment_20av():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config20e"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (normal rff), n_samples=1, Deepset 512 simple decoder niter 7 dropout 0.25"
    return output_dict

def Experiment_20bv():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config20e"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (normal rff), n_samples=1, Deepset 512 simple decoder niter 7 dropout 0.25"
    return output_dict

def Experiment_20cv():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config20e"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (normal rff), n_samples=1, Deepset 512 simple decoder niter 7 dropout 0.25"
    return output_dict

def Experiment_20dv():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config20eii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (normal rff), n_samples=1, Deepset 512 simple decoder niter 7 dropout 0.25"
    return output_dict

def Experiment_20avi():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config20f"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (normal rff), n_samples=1, Deepset 512 simple decoder niter 7 dropout 0.1"
    return output_dict

def Experiment_20bvi():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config20f"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (normal rff), n_samples=1, Deepset 512 simple decoder niter 7 dropout 0.1"
    return output_dict

def Experiment_20cvi():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config20f"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (normal rff), n_samples=1, Deepset 512 simple decoder niter 7 dropout 0.1"
    return output_dict

def Experiment_20dvi():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config20fii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (normal rff), n_samples=1, Deepset 512 simple decoder niter 7 dropout 0.1"
    return output_dict

def Experiment_20avii():
    output_dict = Experiment_19a()
    output_dict["config_name"] = "config20g"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with RBF kernel (normal rff), n_samples=1, Deepset 512 simple decoder niter 7 dropout 0.05"
    return output_dict

def Experiment_20bvii():
    output_dict = Experiment_19b()
    output_dict["config_name"] = "config20g"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Periodic Kernel (normal rff), n_samples=1, Deepset 512 simple decoder niter 7 dropout 0.05"
    return output_dict

def Experiment_20cvii():
    output_dict = Experiment_19c()
    output_dict["config_name"] = "config20g"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with Weakly Matern Kernel (normal rff), n_samples=1, Deepset 512 simple decoder niter 7 dropout 0.053"
    return output_dict

def Experiment_20dvii():
    output_dict = Experiment_19d()
    output_dict["config_name"] = "config20gii"
    output_dict["other"]["info"] = "New rff trial with params matched with ConvCNP paper with sawtooth (normal rff), n_samples=1, Deepset 512 simple decoder niter 7 dropout 0.05"
    return output_dict


### Experiment 21
def Experiment_21a():
    return dict(
        config_name = "config21",
        learner = dict(
            learner = GPLearner,
            model = MetaFunRegressorV4,
            load_fn = GPLearnerLoad,
            model_name = "MetaFunRegressorV4",
        ),
        train_fn = GPTrain,
        test_fn = GPTest,
        data = dict( # for data loading function parser in run.py
            load_fn = GPDataLoad,
            dataprovider = gp_provider,
            load_type = "single",
            custom_kernels = {"RBF_Kernel2":kernels.RBF(length_scale=(0.25))}, 
            custom_kernels_merge = False, 
        ),
        other = dict( # for saving
            info = "Deep kernel experiment",
        )
        )

def Experiment_21b():
    output_dict = Experiment_21a()
    output_dict["data"]["custom_kernels"] = {"Weakly_Periodic_Kernel": kernels.ExpSineSquared(length_scale=1., periodicity=0.25) * kernels.RBF(length_scale=0.5)}
    output_dict["other"]["info"] = "Deep kernel experiment"
    return output_dict

def Experiment_21c():
    output_dict = Experiment_21a()
    output_dict["data"]["custom_kernels"] = {"Matern_Kernel": kernels.Matern(length_scale=0.25, nu=2.5)}
    output_dict["other"]["info"] = "Deep kernel experiment"
    return output_dict

def Experiment_21d():
    output_dict = Experiment_21a()
    output_dict["data"]["load_type"] = "sawtooth"
    output_dict["config_name"] = "config21ii"
    output_dict["other"]["info"] = "Deep kernel experiment"
    return output_dict

def Experiment_21dii():
    output_dict = Experiment_21a()
    output_dict["data"]["load_type"] = "sawtooth"
    output_dict["config_name"] = "config21iib"
    output_dict["other"]["info"] = "Deep kernel experiment with increased stddev to 0.01"
    return output_dict

def Experiment_22a():
    output_dict = Experiment_21a()
    output_dict["config_name"] = "config22"
    output_dict["other"]["info"] = "rff experiment"
    return output_dict

def Experiment_22b():
    output_dict = Experiment_21b()
    output_dict["config_name"] = "config22"
    output_dict["other"]["info"] = "rff experiment"
    return output_dict

def Experiment_22c():
    output_dict = Experiment_21c()
    output_dict["config_name"] = "config22"
    output_dict["other"]["info"] = "rff experiment"
    return output_dict

def Experiment_22d():
    output_dict = Experiment_21d()
    output_dict["config_name"] = "config22ii"
    output_dict["other"]["info"] = "rff experiment"
    return output_dict

def Experiment_23a():
    output_dict = Experiment_21a()
    output_dict["config_name"] = "config23"
    output_dict["other"]["info"] = "Deep Kernel indp iter"
    return output_dict

def Experiment_23b():
    output_dict = Experiment_21b()
    output_dict["config_name"] = "config23"
    output_dict["other"]["info"] = "Deep Kernel indp iter"
    return output_dict

def Experiment_23c():
    output_dict = Experiment_21c()
    output_dict["config_name"] = "config23"
    output_dict["other"]["info"] = "Deep Kernel indp iter"
    return output_dict

def Experiment_23d():
    output_dict = Experiment_21d()
    output_dict["config_name"] = "config23ii"
    output_dict["other"]["info"] = "Deep kernel indp iter with increased stddev to 0.01"
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