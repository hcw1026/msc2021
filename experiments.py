from data.gp_regression import DataProvider as gp_provider
from data.leo_imagenet import DataProvider as imagenet_provider
from learner import ImageNetLearner, GPLearner
from model import MetaFunClassifier, MetaFunRegressor
from sklearn.gaussian_process import kernels
from run import GPTrain, GPTest, GPLearnerLoad, GPDataLoad, ImageNetTrain, ImageNetTest, ImageNetLearnerLoad, ImageNetDataLoad

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
            info = "Simple MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, parametric init, deep-se kernel",
        )
        )

def Experiment_1bii():

    return dict(
        config_name = "config1b",
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
            info = "Simple MetaFunRegressor with Periodic Kernel, with decoder, same neural iteration, parametric init, deep-se kernel, with lower lr than Experiment 2",
        )
        )

def Experiment_1c():

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
            info = "Simple MetaFunRegressor with Noisy Matern Kernel, with decoder, same neural iteration, parametric init, deep-se kernel",
        )
        )

def Experiment_1d():

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
            info = "Simple MetaFunRegressor with Variable Matern Kernel, with decoder, same neural iteration, parametric init, deep-se kernel",
        )
        )

def Experiment_1e():

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
            info = "Simple MetaFunRegressor with a combination of RBF, Periodic and Noisy Matern, with decoder, same neural iteration, parametric init, deep-se kernel",
        )
        )


def Experiment_cls1(): #imagenet experiments

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
            info = "MetaFunClassifier on tieredimagenet with deep se kernel and neural update",
        )
        )


def Experiment_cls1b(): #imagenet experiments

    return dict(
        config_name = "config_cls1b",
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
            info = "MetaFunClassifier on tieredimagenet with attention and neural update",
        )
        )


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