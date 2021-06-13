import tensorflow as tf
import sonnet as snt

class MetaFunClassifier(snt.Module):
    def __init__(self, config, name="MetaFunClassifier"):
        '''
        config: dict
            Configuation python dictionary, see ./config/sample.yaml
        name: str
            Name of classifier
        '''
        super(MetaFunClassifier,self).__init__(name=name)
        self._float_dtype = tf.float32
        self._int_dtype = tf.int32

        # Parse configurations
        ## Components configurations
        _config = config["Model"]["comp"]
        self._use_kernel = _config["use_kernel"]
        self._use_gradient = _config["use_gradient"]
        self._attention_type = tf.constant(_config["attention_type"],dtype=tf.string)
        self._kernel_type = tf.constant(_config["kernel_type"],dtype=tf.string)
        self._no_decoder = _config["no_decoder"]
        self._initial_state_type = tf.constant(_config["initial_state_type"],dtype=tf.string)

        ## Architecture configurations
        _config = config["Model"]["arch"]
        self._nn_size = _config["nn_size"]
        self._nn_layers = _config["nn_layers"]
        self._dim_reprs = _config["dim_reprs"]
        self._num_iters = _config["num_iters"]
        self._embedding_layers = _config["embedding_layers"]

        # Regularisation configurations
        _config = config["Model"]["reg"]
        self._l2_penalty_weight = _config["l2_penalty_weight"]
        self._dropout_rate = _config["dropout_rate"]
        self._label_smoothing = _config["label_smoothing"]
        self._orthogonality_penalty_weight = _config["orthogonality_penalty_weight"]

        # Data configurations
        self._num_classes = config["Data"]["num_classes"]

        # Other configurations
        self._initial_inner_lr = config["Model"]["other"]["initial_inner_lr"]
        self._nonlinearity = tf.nn.relu

        if self._no_decoder:
            self._dim_reprs = 1

    def __call__(self, data, is_training=True):
        '''
        data: dictionary-like form, with attributes "tr_input", "val_input", "tr_output", "val_output"
            Classification training/validation data of a task.
        is_training: bool
            If True, training mode
        '''

        # Initialise
        self.is_training = tf.constant(is_training,dtype=tf.bool)
        self.embedding_dim = data.tr_input.get_shape()[-1]

        # Setup data
        tr_input = data.tr_input
        val_input = data.val_input
        tr_output = data.tr_output
        val_output = data.val_output

        # Initialise
        tr_reprs = self.forward_initialiser(data.tr_input)
        val_reprs = self.forward_initialiser(data.val_input)



    def forward_initialiser(self, x):
        num_points = tf.shape(x)[0]
        if self._initial_state_type == tf.constant("zero",dtype=tf.string):
            reprs = self.constant_initialiser(num_points,trainable=False)
        elif self._initial_state_type == tf.constant("constant",dtype=tf.string):
            reprs = self.constant_initialiser(num_points,trainable=True)
        elif self._initial_state_type == tf.constant("parametric",dtype=tf.string):
            reprs = self.parametric_initialiser(x)
        else:
            raise NameError("Unknown initial state type")
        return reprs

    def constant_initialiser(self, num_points, trainable=False):
        if trainable:
            init = tf.Variable(
                tf.constant_initializer(0.)(
                    shape=[1, self._dim_reprs],
                    dtype = self._float_dtype),
                    trainable=True,
                    name="initial_state")

        else:
            init = tf.zeros([1,self._dim_reprs])

        init = tf.tile(init,[num_points, self._num_classes])
        return init

    def parametric_initialiser(self, x):
        after_dropout = tf.nn.dropout(x, rate=self._dropout_rate)
        initializer = tf.keras.initializers.GlorotUniform() #dtype depends on input #TODO: cast input
        module = snt.nets.MLP(
            output_sizes=[self._nn_size] * self._nn_layers + [self._dim_reprs],
            w_init = initializer,
            with_bias=True,
            activation=self._nonlinearity,
            name="parametric_initialiser"
        )
        outputs = snt.BatchApply(module, num_dims=1)(after_dropout)
        outputs = tf.tile(outputs,[1,self._num_classes])
        return outputs



if __name__ == "__main__":
    from utils import parse_config
    import os
    import numpy as np
    import collections
    config = parse_config(os.path.join(os.path.dirname(__file__),"config/debug.yaml"))
    module = MetaFunClassifier(config=config)

    ClassificationDescription = collections.namedtuple(
    "ClassificationDescription",
    ["tr_input", "tr_output", "val_input", "val_output"])
    
    data = ClassificationDescription(
    tf.constant(np.random.random([10,10]),dtype=tf.float32),
    tf.constant(np.random.uniform(1,10,10).reshape(1,-1),dtype=tf.int32),
    tf.constant(np.random.random([10,10]),dtype=tf.float32),
    tf.constant(np.random.uniform(1,10,10).reshape(1,-1),dtype=tf.int32))

    module(data)

        





        