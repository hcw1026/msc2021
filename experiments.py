from data.gp_regression import DataProvider as gp_provider
from data.leo_imagenet import DataProvider as imagenet_provider
from learner import ImageNetLearner, GPLearner
from model import MetaFunClassifier, MetaFunRegressor
from sklearn.gaussian_process import kernels
from run import GPTrain, GPTest, GPLearnerLoad, GPDataLoad

def Experiment_1():

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
            info = "Simple MetaFunRegressor with RBF Kernel",
        )
        )

def Experiment_2():

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
            custom_kernels = {"Periodic_Kernel":kernels.ExpSineSquared(length_scale=0.5, periodicity=0.5)}, 
            custom_kernels_merge = False, 
        ),
        other = dict( # for saving
            info = "Simple MetaFunRegressor with Periodic Kernel",
        )
        )

def Experiment_3():

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
            custom_kernels = {"Noisy_Matern_Kernel":kernels.WhiteKernel(noise_level=0.1) + kernels.Matern(length_scale=0.2, nu=1.5)}, 
            custom_kernels_merge = False, 
        ),
        other = dict( # for saving
            info = "Simple MetaFunRegressor with Noisy Matern Kernel",
        )
        )

def Experiment_4():

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
            load_type = "var_hyp",
            custom_kernels = {"Variable_Matern_Kernel":kernels.Matern(length_scale_bounds=(0.01, 0.3), nu=1.5)}, 
            custom_kernels_merge = False, 
        ),
        other = dict( # for saving
            info = "Simple MetaFunRegressor with Variable Matern Kernel",
        )
        )

def Experiment_5():

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
            load_type = "var_kernel",
            custom_kernels = {"RBF_Kernel":kernels.RBF(length_scale=(0.2)), "Periodic_Kernel":kernels.ExpSineSquared(length_scale=0.5, periodicity=0.5), "Noisy_Matern_Kernel":kernels.WhiteKernel(noise_level=0.1) + kernels.Matern(length_scale=0.2, nu=1.5)}, 
            custom_kernels_merge = True, 
        ),
        other = dict( # for saving
            info = "Simple MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern",
        )
        )