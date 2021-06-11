import tensorflow as tf
import numpy as np
import sonnet as snt

class MetaFunClassifier(snt.Module):
    def __init__(self,config,name="MetaFunClassifier"):
        super(MetaFunClassifier,self).__init__(name=name)
        self._float_dtype = tf.float32
        self._int_dtype = tf.int32

        # Parse configurations
        ## Components configurations
        _config = config["Model"]["comp"]
        self._use_kernel = _config["use_kernel"]
        self._use_gradient = _config["use_gradient"]
        self._attention_type = _config["attention_type"]
        self._kernel_type = _config["kernel_type"]
        self._no_decoder = _config["no_decoder"]
        self._initial_state_type = _config["initial_state_type"]

        ## Architecture configurations
        _config = config["Model"]["arch"]
        self._nn_size = _config["nn_size"]
        self._nn_layers = _config["nn_layers"]
        self._dim_reprs = _config["dim_reprs"]
        self._num_iters = _config["num_iters"]
        self._embedding_layers = _config["embedding_layers"]

        #Regularisation configurations
        _config = config["Model"]["reg"]
        self._l2_penalty_weight = _config["l2_penalty_weight"]
        self._dropout_rate = _config["dropout_rate"]
        self._label_smoothing = _config["label_smoothing"]
        self._orthogonality_penalty_weight = _config["orthogonality_penalty_weight"]

        #Data configurations
        self._num_classes = config["data"]["num_classes"]

        #Other configurations
        self._initial_inner_lr = config["Model"]["other"]["num_classes"]
        self._nonlinearity = tf.nn.relu

        if self._no_decoder:
            self._dim_reprs = 1

    def __call__(self, data, is_training=True):
        data = cls_data.ClassificationDescription(*data)



        