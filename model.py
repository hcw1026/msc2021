import tensorflow as tf
import sonnet as snt
import submodules

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
        self._use_kernel = tf.constant(_config["use_kernel"], dtype=tf.bool)
        self._use_gradient = tf.constant(_config["use_gradient"], dtype=tf.bool)
        self._attention_type = tf.constant(_config["attention_type"], dtype=tf.string)
        self._kernel_type = tf.constant(_config["kernel_type"], dtype=tf.string)
        self._no_decoder = tf.constant(_config["no_decoder"], dtype=tf.bool)
        self._initial_state_type = tf.constant(_config["initial_state_type"], dtype=tf.string)

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
        self._dropout_rate = tf.constant(_config["dropout_rate"], dtype=self._float_dtype)
        self._label_smoothing = _config["label_smoothing"]
        self._orthogonality_penalty_weight = _config["orthogonality_penalty_weight"]

        # Data configurations
        self._num_classes = config["Data"][data_source]["num_classes"]

        # Other configurations
        self._initial_inner_lr = config["Model"]["other"]["initial_inner_lr"]
        self._nonlinearity = tf.nn.relu
        self.initialiser = tf.keras.initializers.GlorotUniform()
        self.dropout = tf.keras.layers.Dropout(self._dropout_rate)

        # Constant Initialization
        self._orthogonality_reg = tf.constant(0., dtype=self._float_dtype)
        self.is_training = tf.constant(True, dtype=tf.bool)
        self.embedding_dim = tf.constant([1])

        # Metric initialisation
        self.metric = tf.keras.metrics.BinaryAccuracy(name="inner accuracy", dtype=self._float_dtype)

        if self._no_decoder:
            self._dim_reprs = 1

    @snt.once
    def _initialize(self):
        """initialiser variables and functions"""
        # Inner learning rate
        self.alpha =  tf.Variable(
            initial_value=tf.constant_initializer(self._initial_inner_lr)(
                shape=[1,1],
                dtype=self._float_dtype
            ),
            trainable=True,
            name="alpha"
            )


        # Forward initialiser
        # self.forward_initialiser = submodules.forward_initialiser(
        #     initial_state_type=self._initial_state_type,
        #     dim_reprs=self._dim_reprs,
        #     num_classes=self._num_classes,
        #     nn_size=self._nn_size,
        #     nn_layers=self._nn_layers,
        #     float_dtype=self._float_dtype,
        #     dropout_rate=self._dropout_rate,
        #     initialiser=self.initialiser,
        #     nonlinearity=self._nonlinearity)
        self.forward_initialiser = self._forward_initialiser()


        # Forward local updater
        if not self._use_gradient:
            self._neural_local_updater = submodules.neural_local_updater(
                nn_size=self._nn_size, 
                nn_layers=self._nn_layers, 
                dim_reprs=self._dim_reprs, 
                num_classes=self._num_classes, 
                initialiser=self.initialiser, 
                nonlinearity=self._nonlinearity)
        
        else:
            # Inner learning rate for each functional representation for each class
            self.lr = tf.Variable(
                initial_value=tf.constant_initializer(1.0)(
                    shape=[1,self._num_classes * self._dim_reprs],
                    dtype=self._float_dtype,
                ),
                trainable=True,
                name="lr"
            )

        self.forward_local_updater = self._forward_local_updater()

        # Forward decoder
        if not self._no_decoder:
            self.decoder = submodules.decoder(
                embedding_dim = self.embedding_dim,
                orthogonality_penalty_weight = self._orthogonality_penalty_weight, 
                initialiser = self.initialiser)

        self.forward_decoder = self._forward_decoder()


        # Kernel or attention
        if self._use_kernel:
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
        
            if self._kernel_type == tf.constant("se", dtype=tf.string):
                pass
            elif self._kernel_type == tf.constant("deep_se", dtype=tf.string):
                self.deep_se_kernel = submodules.deep_se_kernel(
                    embedding_layers=self._embedding_layers,
                    kernel_dim=self.embedding_dim,
                    initialiser=self.initialiser,
                    nonlinearity=self._nonlinearity
                )
            else:
                raise NameError("Unknown kernel type")
        else:
            config = {
                "rep": "mlp",
                "output_sizes": [self.embedding_dim] * self._embedding_layers,
                "att_type": self._attention_type,
                "normalise": tf.constant(True, tf.bool),
                "scale": 1.0,
                "l2_penalty_weight": self._l2_penalty_weight,
                "nonlinearity": self._nonlinearity
                }

            self.attention = submodules.Attention(config)

        self.forward_kernel_or_attention = self._forward_kernel_or_attention()


    def __call__(self, data, is_training=True):
        """
        data: dictionary-like form, with attributes "tr_input", "val_input", "tr_output", "val_output"
            Classification training/validation data of a task.
        is_training: bool
            If True, training mode
        """

        # Initialise Variables #TODO: is_training set as tf.constant in learner
        self.embedding_dim = data.tr_input.get_shape()[-1]
        self._initialize()

        # Initialise r
        tr_reprs = self.forward_initialiser(data.tr_input, is_training=is_training)
        val_reprs = self.forward_initialiser(data.val_input, is_training=is_training)

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

    # def forward_local_updater(self, r, y, x=None, iter=""):
    #     """functional representation updater"""
    #     if self._use_gradient:
    #         updates = self.gradient_local_updater(r=r, y=y, x=x, iter=iter)
    #     else:
    #         r_shape = r.shape.as_list()
    #         r = tf.reshape(r, r_shape[:-1] + [self._num_classes, self._dim_reprs])
    #         updates = self._neural_local_updater(r=r, y=y, x=x, iter=iter)
    #         updates = tf.reshape(updates, shape=r_shape)
    #     return updates

    def _forward_initialiser(self):
        """initialise neural latent - r"""
        if self._initial_state_type == tf.constant("zero", dtype=tf.string):
            self.constant_initialiser = submodules.constant_initialiser(
                dim_reprs=self._dim_reprs, 
                float_dtype=self._float_dtype, 
                num_classes=self._num_classes, 
                trainable=False)
            return lambda x: self.constant_initialiser(x.shape[-1])
        elif self._initial_state_type == tf.constant("constant", dtype=tf.string):
            self.constant_initialiser = submodules.constant_initialiser(
                dim_reprs=self._dim_reprs, 
                float_dtype=self._float_dtype, 
                num_classes=self._num_classes, 
                trainable=True)
            return lambda x: self.constant_initialiser(x.shape[-1])
        elif self._initial_state_type == tf.constant("parametric", dtype=tf.string):
            self.parametric_initialiser = submodules.parametric_initialiser(
                nn_size=self._nn_size,
                nn_layers=self._nn_layers,
                dim_reprs=self._dim_reprs,
                num_classes=self._num_classes,
                dropout_rate=self._dropout_rate,
                initialiser=self.initialiser,
                nonlinearity=self._nonlinearity,
            )
            return self.parametric_initialiser
        else:
            raise NameError("Unknown initial state type")


    def _forward_local_updater(self):
        """functional representation updater"""
        if self._use_gradient:
            return self.gradient_local_updater
        else:
            return self.neural_local_updater

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

    def neural_local_updater(self, r, y, x, iter=""):
        r_shape = r.shape.as_list()
        r = tf.reshape(r, r_shape[:-1] + [self._num_classes, self._dim_reprs])
        updates = self._neural_local_updater(r=r, y=y, x=x, iter=iter)
        updates = tf.reshape(updates, shape=r_shape)
        return updates

    # def _forward_decoder(self, cls_reprs):
    #     """decode and sample weight for final outpyt layer"""
    #     if self._no_decoder: 
    #         # use functional representation directly as the predictor, used in ablation study
    #         return cls_reprs
    #     else:
    #         s = cls_reprs.shape.as_list()
    #         cls_reprs = tf.reshape(cls_reprs, s[:-1] + [self._num_classes, self._dim_reprs]) # split each representation into classes
    #         weights_dist_params, self._orthogonality_reg = self.decoder(cls_reprs) # get mean and variance of wn in LEO
    #         stddev_offset = tf.math.sqrt(2. / (self.embedding_dim + self._num_classes)) #from LEO
    #         classifier_weights = self.sample(
    #             distribution_params=weights_dist_params,
    #             stddev_offset=stddev_offset)
    #         return classifier_weights

    def _forward_decoder(self):
        """decode and sample weight for final outpyt layer"""
        if self._no_decoder: 
            # use functional representation directly as the predictor, used in ablation study
            return lambda x: x
        else:
            return self._forward_decoder_with_decoder

    def _forward_decoder_with_decoder(self, cls_reprs):
        s = cls_reprs.shape.as_list()
        cls_reprs = tf.reshape(cls_reprs, s[:-1] + [self._num_classes, self._dim_reprs]) # split each representation into classes
        weights_dist_params, self._orthogonality_reg = self.decoder(cls_reprs) # get mean and variance of wn in LEO
        stddev_offset = tf.math.sqrt(2. / (self.embedding_dim + self._num_classes)) #from LEO
        classifier_weights = self.sample(
            distribution_params=weights_dist_params,
            stddev_offset=stddev_offset)
        return classifier_weights

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
        else:
            after_dropout = self.dropout(inputs, training=self.is_training)
            preds = tf.linalg.matvec(weights, after_dropout) # (x^Tw)_k - i=instance, m=class, k=features
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


    # def forward_kernel_or_attention(self, querys, keys, values):
    #     """functional pooling"""
    #     if self._use_kernel:
    #         if self._kernel_type == tf.constant("se", dtype=tf.string):
    #             rtn_values = submodules.squared_exponential_kernel(querys, keys, values, self.sigma, self.lengthscale)
    #         else:
    #             rtn_values = self.deep_se_kernel(querys, keys, values, self.sigma, self.lengthscale)

    #     else:
    #         rtn_values = self.attention_block(querys, keys, values)

    #     return rtn_values

    def _forward_kernel_or_attention(self):
        """functional pooling"""
        if self._use_kernel:
            if self._kernel_type == tf.constant("se", dtype=tf.string):
                return lambda querys, keys, values: submodules.squared_exponential_kernel(querys, keys, values, self.sigma, self.lengthscale)
            else:
                return lambda querys, keys, values: self.deep_se_kernel(querys, keys, values, self.sigma, self.lengthscale)
        else:
            return self.attention_block

    def attention_block(self, querys, keys, values):
        """dot-product kernel"""
        return self.attention(keys, querys, values)

    @property
    def _decoder_orthogonality_reg(self):
        return self._orthogonality_reg

if __name__ == "__main__":
    from utils import parse_config
    import os
    import numpy as np
    import collections
    config = parse_config(os.path.join(os.path.dirname(__file__),"config/debug.yaml"))
    module = MetaFunClassifier(config=config)

    # ClassificationDescription = collections.namedtuple(
    # "ClassificationDescription",
    # ["tr_input", "tr_output", "val_input", "val_output"])
    
    # data = ClassificationDescription(
    # tf.constant(np.random.random([10,10]),dtype=tf.float32),
    # tf.constant(np.random.uniform(1,10,10).reshape(-1,1),dtype=tf.int32),
    # tf.constant(np.random.random([10,10]),dtype=tf.float32),
    # tf.constant(np.random.uniform(1,10,10).reshape(-1,1),dtype=tf.int32))

    # print(module(data))
    from data.leo_imagenet import DataProvider
    dataloader = DataProvider("trial", config)
    dat = dataloader.generate()
    for i in dat.take(1):
        module(i)

    @tf.function
    def trial(x):
        l,_,_ = module(x)
        return l
    print("DEBUGGGGGGGGGGGGGGG")
    for i in dat.take(1):
        trial(i)