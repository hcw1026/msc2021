import tensorflow as tf
import sonnet as snt

class MetaFunClassifier(snt.Module):
    def __init__(self, config, data_source="leo_imagenet", name="MetaFunClassifier"):
        """
        config: dict
            Configuation python dictionary, see ./config/sample.yaml
        data_source: str
            The sub-level name within the data level of config used for the problem
        name: str
            Name of classifier
        """
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
        self._num_classes = config["Data"][data_source]["num_classes"]

        # Other configurations
        self._initial_inner_lr = config["Model"]["other"]["initial_inner_lr"]
        self._nonlinearity = tf.nn.relu
        self.initializer = tf.keras.initializers.GlorotUniform()

        # Constant Initialization
        self._orthogonality_reg = tf.constant(0.,dtype=self._float_dtype)
        self.is_training = tf.constant(True,dtype=tf.bool)
        self.embedding_dim = tf.constant([1])

        # Metric initialisation
        self.metric = tf.keras.metrics.BinaryAccuracy(name="inner accuracy", dtype=self._float_dtype)

        if self._no_decoder:
            self._dim_reprs = 1

    @snt.once
    def _initialize(self):
        """initialiser variables"""
        # Inner learning rate
        self.alpha =  tf.Variable(
            initial_value=tf.constant_initializer(self._initial_inner_lr)(
                shape=[1,1],
                dtype=self._float_dtype
            ),
            trainable=True,
            name="alpha"
            )

        # Inner learning rate for each functional representation for each class
        self.lr = tf.Variable(
            initial_value=tf.constant_initializer(1.0)(
                shape=[1,self._num_classes * self._dim_reprs],
                dtype=self._float_dtype,
            ),
            trainable=True,
            name="lr"
        )

        # Kernel sigma
        self.sigma = tf.Variable(
            initial_value=tf.constant_initializer(1.0)(
                shape=(),
                dtype=self._float_dtype
            ),
            trainable=True,
            name="kernel_sigma"
        )

        # Kernel lengthscale
        self.lengthscale = tf.Variable(
            initial_value=tf.constant_initializer(1.0)(
                shape=(),
                dtype=self._float_dtype
            ),
            trainable=True,
            name="kernel_lengthscale"
        )

    def __call__(self, data, is_training=True):
        """
        data: dictionary-like form, with attributes "tr_input", "val_input", "tr_output", "val_output"
            Classification training/validation data of a task.
        is_training: bool
            If True, training mode
        """

        # Initialise
        self._initialize()
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

        # Iterative functional updating
        for k in range(self._num_iters):
            updates = self.forward_local_updater(tr_reprs, data.tr_output, data.tr_input) #return negative u
            tr_updates = self.alpha * self.forward_kernel_or_attention(querys=data.tr_input, keys=data.tr_input, values=updates)
            val_updates = self.alpha * self.forward_kernel_or_attention(querys=data.val_input, keys=data.tr_input, values=updates)
            tr_reprs += tr_updates
            val_reprs += val_updates

        # Decode functional representation and compute loss and metric
        classifier_weights = self.forward_decoder(tr_reprs)
        tr_loss, batch_tr_metric = self.calculate_loss_and_acc(data.tr_input, data.tr_output, classifier_weights)
        classifier_weights = self.forward_decoder(val_reprs)
        val_loss, batch_val_metric = self.calculate_loss_and_acc(data.val_input, data.val_output, classifier_weights)

        # Aggregate loss in a batch
        batch_tr_loss = tf.math.reduce_mean(tr_loss)
        batch_val_loss = tf.math.reduce_mean(val_loss)

        #Additional regularisation penalty
        return batch_val_loss + self._decoder_orthogonality_reg, batch_tr_metric, batch_val_metric  #TODO:? need weights for l2

    def forward_initialiser(self, x):
        """functional representation initialisation"""
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
        """functional representation initialiser - constant"""
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
        """functional representation intialiser - parametric"""
        after_dropout = tf.nn.dropout(x, rate=self._dropout_rate)
        module = snt.nets.MLP( # dtype depends on input #TODO: cast input
            output_sizes=[self._nn_size] * self._nn_layers + [self._dim_reprs],
            w_init = self.initializer,
            with_bias=True,
            activation=self._nonlinearity,
            name="parametric_initialiser"
        )
        outputs = snt.BatchApply(module, num_dims=1)(after_dropout)
        outputs = tf.tile(outputs,[1,self._num_classes])
        return outputs

    def forward_local_updater(self, r, y, x=None, iter=""):
        """functional representation updater"""
        if self._use_gradient:
            updates = self.gradient_local_updater(r=r, y=y, x=x, iter=iter)
        else:
            r_shape = r.shape.as_list()
            r = tf.reshape(r, r_shape[:-1] + [self._num_classes, self._dim_reprs])
            updates = self.neural_local_updater(r=r, y=y, x=x, iter=iter)
            updates = tf.reshape(updates, shape=r_shape)
        return updates

    def gradient_local_updater(self, r, y, x=None, iter=""):
        """functional gradient update instead of neural update"""
        with tf.GradientTape() as tape:
            tape.watch(r) #watch r
            classifier_weights = self.forward_decoder(r) # sample w_n from LEO
            tr_loss, _ = self.calculate_loss_and_acc(x, y, classifier_weights) # softmax cross entropy
            batch_tr_loss = tf.reduce_mean(tr_loss)
        
        loss_grad = tape.gradient(batch_tr_loss, r)
        updates = - self.lr * loss_grad
        return updates

    def neural_local_updater(self, r, y, x=None, iter=""):
        """functional neural update"""
        y = tf.one_hot(y, self._num_classes) #TODO: move to data preprocessing
        y = tf.transpose(y, perm=[0, 2, 1])

        # MLP m
        module1 = snt.nets.MLP(
            output_sizes=[self._nn_size] * self._nn_layers,
            w_init=self.initializer,
            with_bias=True,
            activation=self._nonlinearity,
            name="neural_local_updater_m"
        )
        outputs = snt.BatchApply(module1, num_dims=2)(r)
        agg_outputs = tf.math.reduce_mean(outputs, axis=-2, keepdims=True)
        outputs = tf.concat([outputs, tf.tile(agg_outputs, [1, self._num_classes,1])], axis=-1)

        #MLP u+
        module2 = snt.nets.MLP(
            output_sizes=[self._nn_size] * self._nn_layers,
            w_init=self.initializer,
            with_bias=True,
            activation=self._nonlinearity,
            name="neural_local_updater_uplus"
        )
        outputs_t = snt.BatchApply(module2, num_dims=2)(outputs)

        #MLP u-
        module3 = snt.nets.MLP(
            [self._nn_size] * self._nn_layers + [self._dim_reprs],
            w_init=self.initializer,
            with_bias=True,
            activation=self._nonlinearity,
            name="neural_local_updater_uminus"
        )
        outputs_f = snt.BatchApply(module3, num_dims=2)(outputs)
        outputs = outputs_t * y + outputs_f * (1-y)
        return outputs

    def forward_decoder(self, cls_reprs):
        """decode and sample weight for final outpyt layer"""
        if self._no_decoder: 
            # use functional representation directly as the predictor, used in ablation study
            return cls_reprs
        s = cls_reprs.shape.as_list()
        cls_reprs = tf.reshape(cls_reprs, s[:-1] + [self._num_classes, self._dim_reprs]) # split each representation into classes
        weights_dist_params = self.decoder(cls_reprs) # get mean and variance of wn in LEO
        stddev_offset = tf.math.sqrt(2. / (self.embedding_dim + self._num_classes)) #from LEO
        classifier_weights = self.sample(
            distribution_params=weights_dist_params,
            stddev_offset=stddev_offset)
        return classifier_weights

    def decoder(self,inputs):
        """decode functional representation (z)"""
        orthogonality_reg = get_orthogonality_regularizer(
            self._orthogonality_penalty_weight) # final term of regularisation from LEO
        # 2 * embedding_dim, because both means and variances are returned
        decoder_module = snt.Linear(
            output_size=self.embedding_dim * 2,
            with_bias=True,
            w_init=self.initializer,
            name="decoder"
        )
        outputs = snt.BatchApply(decoder_module,num_dims=2)(inputs) # w in LEO
        self._orthogonality_reg = orthogonality_reg(decoder_module.w) # .w get weights
        return outputs

    def sample(self, distribution_params, stddev_offset=0.):
        """sample from a normal distribution"""
        means, unnormalized_stddev = tf.split(distribution_params, 2, axis=-1) # mean and log std
        stddev = tf.math.exp(unnormalized_stddev) - (1. - stddev_offset)
        stddev = tf.math.maximum(stddev, 1e-10)
        if not self.is_training:
            return means
        else:
            return tf.random.normal(shape=means.shape, mean=means, stddev=stddev)

    def calculate_loss_and_acc(self, inputs, true_outputs, classifier_weights):
        """compute cross validation loss"""
        model_outputs = self.predict(inputs, classifier_weights) # return unnormalised probability of each class for each instance
        model_predictions = tf.math.argmax(
            input=model_outputs, axis=-1, output_type=self._int_dtype)
        accuracy = self.metric(model_predictions, tf.squeeze(true_outputs, axis=-1))
        self.metric.reset_state()
        return self.loss_fn(model_outputs, true_outputs), accuracy

    def predict(self, inputs, weights):
        """unnormalised class probabilities"""
        if self._no_decoder:
            return weights
        after_dropout = tf.nn.dropout(inputs, rate=self._dropout_rate)
        preds = tf.linalg.einsum("ik,imk->im", after_dropout, weights) # (x^Tw)_k - i=instance, m=class, k=features
        return preds

    def loss_fn(self, model_outputs, original_classes):
        """binary cross entropy"""
        original_classes = tf.squeeze(original_classes, axis=-1)
        one_hot_outputs = tf.one_hot(original_classes, depth=self._num_classes) #TODO: move onehot to data preprocessing
        return tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            label_smoothing=self._label_smoothing,
            reduction=tf.keras.losses.Reduction.NONE)(
                y_true=one_hot_outputs,
                y_pred=model_outputs)

    def forward_kernel_or_attention(self, querys, keys, values):
        """functional pooling"""
        if self._use_kernel:
            if self._kernel_type == tf.constant("se", dtype=tf.string):
                rtn_values = self.squared_exponential_kernel(querys, keys, values)
            elif self._kernel_type == tf.constant("deep_se", dtype=tf.string):
                rtn_values = self.deep_se_kernel(querys, keys, values)
            else:
                raise NameError("Unknown kernel type")
        else:
            rtn_values = self.attention_block(querys, keys, values)

        return rtn_values

    def squared_exponential_kernel(self, querys, keys, values):
        """rbf kernel"""
        num_keys = tf.shape(keys)[0]
        num_querys = tf.shape(querys)[0]

        _keys = tf.tile(tf.expand_dims(keys, axis=1), [1, num_querys, 1])
        _querys = tf.tile(tf.expand_dims(querys, axis=0), [num_keys, 1, 1])
        sq_norm = tf.reduce_sum((_keys - _querys)**2, axis=-1)
        kernel_qk = self.sigma**2 * tf.math.exp(- sq_norm / (2.*self.lengthscale**2)) # RBF
        v = tf.linalg.matmul(kernel_qk, values, transpose_a=True) # RBF * V
        return v


    def deep_se_kernel(self, querys, keys, values):
        """deep rbf kernel"""
        module = snt.nets.MLP( # mapping a
            output_sizes=[self.embedding_dim] * self._embedding_layers,
            w_init=self.initializer,
            with_bias=True,
            activation=self._nonlinearity,
            name="deep_se_kernel"
        )
        keys = snt.BatchApply(module, num_dims=1)(keys)
        querys = snt.BatchApply(module, num_dims=1)(querys)

        return self.squared_exponential_kernel(querys, keys, values)

    def attention_block(self, querys, keys, values):
        """dot-product kernel"""
        config = {
            "rep": "mlp",
            "output_sizes": [self.embedding_dim] * self._embedding_layers,
            "att_type": self._attention_type,
            "normalise": tf.constant(True,tf.bool),
            "scale": 1.0,
            "l2_penalty_weight": self._l2_penalty_weight,
            "nonlinearity": self._nonlinearity
        }

        attention = Attention(config=config)
        v = attention(keys, querys, values)
        return v

    @property
    def _decoder_orthogonality_reg(self):
        return self._orthogonality_reg

    
# (Adapted from https://github.com/deepmind/leo, see copyright and original license in our LICENSE file.)
def get_orthogonality_regularizer(orthogonality_penalty_weight):
    """Returns the orthogonality regularizer."""
    def orthogonality(weight):
        """Calculates the layer-wise penalty encouraging orthogonality."""
        w2 = tf.linalg.matmul(weight, weight, transpose_b=True)
        wn = tf.linalg.norm(weight, ord=2, axis=1, keepdims=True) + 1e-32
        correlation_matrix = w2 / tf.matmul(wn, wn, transpose_b=True)
        matrix_size = correlation_matrix.get_shape().as_list()[0]
        base_dtype = weight.dtype.base_dtype
        identity = tf.linalg.eye(matrix_size,dtype=base_dtype)
        weight_corr = tf.reduce_mean(
            tf.math.squared_difference(correlation_matrix, identity))
        return tf.multiply(
            tf.cast(orthogonality_penalty_weight, base_dtype),
            weight_corr,
            name="orthogonality regularisation"
        )
    return orthogonality


# Attention modules
# (Adapted from https://github.com/deepmind/neural-processes, see copyright and original license in our LICENSE file.)

class Attention(snt.Module):

    def __init__(self, config=None, name="attention"):
        super(Attention, self).__init__(name=name)
        self._float_dtype = tf.float32
        self._int_dtype = tf.int32
        self._rep = config['rep']
        self._output_sizes = config['output_sizes']
        self._att_type = config['att_type']
        self._normalise = config['normalise']
        self._scale = config['scale']
        self._l2_penalty_weight = config['l2_penalty_weight']
        self._nonlinearity = config['nonlinearity']
        self.initializer = tf.keras.initializers.GlorotUniform()

    def __call__(self, x1, x2, r):
        if self._rep == tf.constant("identity", dtype=tf.string):
            k, q = (x1, x2)
        elif self._rep == tf.constant("mlp", dtype=tf.string): # mapping a
            module = snt.nets.MLP( # mapping a
            output_sizes=self._output_sizes,
            w_init=self.initializer,
            with_bias=True,
            activation=self._nonlinearity,
            name="deep_attention"
            )
            k = snt.BatchApply(module, num_dims=1)(x1)
            q = snt.BatchApply(module, num_dims=1)(x2)
        else:
            raise NameError("Unknown attention representation - not among ['identity', 'mlp']")

        if self._att_type == tf.constant("dot_product",tf.string):
            rep = dot_product_attention(q, k, r, self._normalise)
        else:
            raise NameError("Unknown attention type")
        
        return rep

def dot_product_attention(q, k, v, normalise):
    """Computes dot product attention.

    Args:
        q: queries. tensor of  shape [B,m,d_k].
        k: keys. tensor of shape [B,n,d_k].
        v: values. tensor of shape [B,n,d_v].
        normalise: Boolean that determines whether weights sum to 1.
    Returns:
        tensor of shape [B,m,d_v].
    """
    d_k = tf.shape(q)[-1]
    scale = tf.math.sqrt(tf.cast(d_k, tf.float32))
    unnorm_weights = tf.linalg.matmul(q, k, transpose_b=True) / scale # [B,m,n]
    if normalise:
        weight_fn = tf.math.softmax
    else:
        weight_fn = tf.math.sigmoid
    weights = weight_fn(unnorm_weights)
    rep = tf.linalg.matmul(weights, v)
    return rep




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
    tf.constant(np.random.uniform(1,10,10).reshape(-1,1),dtype=tf.int32),
    tf.constant(np.random.random([10,10]),dtype=tf.float32),
    tf.constant(np.random.uniform(1,10,10).reshape(-1,1),dtype=tf.int32))

    print(module(data))

        





        