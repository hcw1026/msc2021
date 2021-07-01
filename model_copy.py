import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt
import submodules_copy as submodules
import utils

class MetaFunBase(snt.Module):
    def __init__(self, config, no_batch, num_classes=1, name="MetaFun"):
        super(MetaFunBase, self).__init__(name=name)

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

        # Other configurations
        self._initial_inner_lr = config["Model"]["other"]["initial_inner_lr"]
        self._nonlinearity = tf.nn.relu
        self.initialiser = tf.keras.initializers.GlorotUniform()
        self.dropout = tf.keras.layers.Dropout(self._dropout_rate)
        self._no_batch = no_batch

        # Constant Initialization
        self._orthogonality_reg = tf.Variable(0., dtype=self._float_dtype, trainable=False)
        self.is_training = tf.Variable(True, dtype=tf.bool, trainable=False)
        self.reset = tf.Variable(True, dtype=tf.bool, trainable=False)
        self.embedding_dim = tf.constant([1])
        self.has_initialised = False

        # Classification or Regression
        self._num_classes = num_classes
        self._repr_as_inputs = config["Model"]["comp"]["reprs_as_inputs"]
        self._neural_updater_concat_x = config["Model"]["comp"]["neural_updater_concat_x"]

        if self._no_decoder:
            self._dim_reprs = 1


    def forward_initialiser_base(self):
        """initialise neural latent - r"""
        if self._initial_state_type == tf.constant("zero", dtype=tf.string):
            self.constant_initialiser_init()
            return self.constant_initialiser
        elif self._initial_state_type == tf.constant("constant", dtype=tf.string):
            self.constant_initialiser_init()
            return self.constant_initialiser
        elif self._initial_state_type == tf.constant("parametric", dtype=tf.string):
            self.parametric_initialiser_init()
            return self.parametric_initialiser
        else:
            raise NameError("Unknown initial state type")

    def forward_local_updater_base(self):
        """functional representation updater"""
        if self._use_gradient:
            self.gradient_local_updater_init()
            return self.gradient_local_updater
        else:
            self.neural_local_updater_init()
            return self.neural_local_updater
    
    def forward_kernel_or_attention_base(self):
        """functional pooling"""
        if self._use_kernel:
            if self._kernel_type == tf.constant("se", dtype=tf.string):
                self.se_kernel_init()
                return self.se_kernel
            elif self._kernel_type == tf.constant("deep_se", dtype=tf.string):
                self.deep_se_kernel_init()
                return self.deep_se_kernel
            else:
                raise NameError("Unknown kernel type")
        else:
            self.attention_block_init()
            return self.attention_block

    def forward_decoder_base(self):
        """decode and sample weight for final outpyt layer"""
        if self._no_decoder: 
            # use functional representation directly as the predictor, used in ablation study
            return lambda x: x
        else:
            self.forward_decoder_with_decoder_init()
            return self.forward_decoder_with_decoder

    @property
    def get_regularise_variables(self):
        return self.regularise_variables


class MetaFunClassifier(MetaFunBase, snt.Module):
    def __init__(self, config, data_source="leo_imagenet", no_batch=False, name="MetaFunClassifier"): #TODO: remove no_batch
        """
        config: dict
            Configuation python dictionary, see ./config/sample.yaml
        data_source: str
            The sub-level name within the data level of config used for the problem
        name: str
            Name of classifier
        """

        # Parse configurations
        # Data configurations
        self._num_classes = config["Data"][data_source]["num_classes"]

        super(MetaFunClassifier, self).__init__(config, no_batch, num_classes=self._num_classes, name=name)

        # Metric initialisation
        self.metric = tf.keras.metrics.BinaryAccuracy(name="inner accuracy", dtype=self._float_dtype)

    @snt.once
    def initialise(self, data_instance):
        """initialiser variables and functions"""
        self.sample_tr_data = data_instance.tr_input
        self.sample_val_data = data_instance.val_input
        self.embedding_dim = self.sample_tr_data.get_shape()[-1]
        
        self.forward_initialiser = self.forward_initialiser_base()
        self.forward_local_updater = self.forward_local_updater_base()
        self.forward_decoder = self.forward_decoder_base()
        self.forward_kernel_or_attention = self.forward_kernel_or_attention_base()

        self.predict = self._predict()

        # Change initialisation state
        self.has_initialised = True

        # Regularisation variables
        self.__call__(data_instance)
        self.regularise_variables = utils.get_linear_layer_variables(self)

    def __call__(self, data, is_training=tf.constant(True, dtype=tf.bool)):
        """
        data: dictionary-like form, with attributes "tr_input", "val_input", "tr_output", "val_output"
            Classification training/validation data of a task.
        is_training: bool
            If True, training mode
        """
        # Initialise Variables #TODO: is_training set as tf.constant in learner
        self.is_training.assign(is_training)
        self.reset.assign(True) # cache precomputed transformation
        assert self.has_initialised, "the learner is not initialised, please initialise via the method - .initialise"

        # Initialise r
        tr_reprs = self.forward_initialiser(data.tr_input, is_training=self.is_training)
        val_reprs = self.forward_initialiser(data.val_input, is_training=self.is_training)
        # Iterative functional updating
        for k in tf.range(self._num_iters):
            updates = self.forward_local_updater(tr_reprs, data.tr_output, data.tr_input) #return negative u
            tr_updates, val_updates = tf.split(
                self.alpha * self.forward_kernel_or_attention(
                    querys=tf.concat([data.tr_input, data.val_input],axis=-2), 
                    keys=data.tr_input, 
                    values=updates, 
                    reset=self.reset), 
                num_or_size_splits=[data.tr_input.shape[-2], data.val_input.shape[-2]], 
                axis=-2)
            tr_reprs += tr_updates
            val_reprs += val_updates
            self.reset.assign(False)

        # Decode functional representation and compute loss and metric
        classifier_weights = self.forward_decoder(tr_reprs)
        tr_loss, batch_tr_metric = self.predict_and_calculate_loss_and_acc(data.tr_input, data.tr_output, classifier_weights)
        classifier_weights = self.forward_decoder(val_reprs)
        val_loss, batch_val_metric = self.predict_and_calculate_loss_and_acc(data.val_input, data.val_output, classifier_weights)


        # # Aggregate loss in a batch
        # batch_tr_loss = tf.math.reduce_mean(tr_loss)
        # batch_val_loss = tf.math.reduce_mean(val_loss)

        #Additional regularisation penalty
        #return batch_val_loss + self._decoder_orthogonality_reg, batch_tr_metric, batch_val_metric  #TODO:? need weights for l2

        return val_loss, batch_tr_metric, batch_val_metric

    @snt.once
    def initialiser_all_init(self):
        self.alpha =  tf.Variable( # Inner learning rate
            initial_value=tf.constant_initializer(self._initial_inner_lr)(
                shape=[1,1],
                dtype=self._float_dtype
            ),
            trainable=True,
            name="alpha"
            )

    @snt.once
    def constant_initialiser_init(self):
        self.initialiser_all_init()
        if self._initial_state_type == tf.constant("zero", dtype=tf.string):
            self._constant_initialiser = submodules.constant_initialiser(
                    dim_reprs=self._dim_reprs, 
                    float_dtype=self._float_dtype, 
                    num_classes=self._num_classes,
                    classification=True, 
                    no_batch=self._no_batch, 
                    trainable=False)
        else:
            self._constant_initialiser = submodules.constant_initialiser(
                    dim_reprs=self._dim_reprs, 
                    float_dtype=self._float_dtype, 
                    num_classes=self._num_classes,
                    classification=True, 
                    no_batch=self._no_batch, 
                    trainable=True)


    def constant_initialiser(self, x, is_training=True):
        return self._constant_initialiser(x)

    @snt.once
    def parametric_initialiser_init(self):
        self.initialiser_all_init()
        self._parametric_initialiser = submodules.parametric_initialiser(
                        nn_size=self._nn_size,
                        nn_layers=self._nn_layers,
                        dim_reprs=self._dim_reprs,
                        num_classes=self._num_classes,
                        classification=True,
                        dropout_rate=self._dropout_rate,
                        initialiser=self.initialiser,
                        nonlinearity=self._nonlinearity,
                    )
    
    def parametric_initialiser(self, x, is_training):
        return self._parametric_initialiser(x, is_training)

    @snt.once
    def gradient_local_updater_init(self):
        # Inner learning rate for each functional representation for each class
        self.lr = tf.Variable(
            initial_value=tf.constant_initializer(1.0)(
                shape=[1,self._num_classes * self._dim_reprs],
                dtype=self._float_dtype,
            ),
            trainable=True,
            name="lr"
        )

    def gradient_local_updater(self, r, y, x=None, iter=""):
        """functional gradient update instead of neural update"""
        with tf.GradientTape() as tape:
            tape.watch(r) #watch r
            classifier_weights = self.forward_decoder(r) # sample w_n from LEO
            tr_loss, _ = self.predict_and_calculate_loss_and_acc(x, y, classifier_weights) # softmax cross entropy
            batch_tr_loss = tf.reduce_mean(tr_loss)

        loss_grad = tape.gradient(batch_tr_loss, r)
        updates = - self.lr * loss_grad
        return updates

    @snt.once
    def neural_local_updater_init(self):
        self._neural_local_updater = submodules.neural_local_updater(
            nn_size=self._nn_size, 
            nn_layers=self._nn_layers, 
            dim_reprs=self._dim_reprs, 
            classification=True,
            num_classes=self._num_classes, 
            initialiser=self.initialiser, 
            nonlinearity=self._nonlinearity,
            no_batch=self._no_batch)

    def neural_local_updater(self, r, y, x, iter=""):
        return self._neural_local_updater(r=r, y=y, x=x, iter=iter)

    @snt.once
    def se_kernel_all_init(self):
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

    @snt.once
    def se_kernel_init(self):
        self.se_kernel_all_init()
        self._se_kernel = submodules.squared_exponential_kernel(
            train_instance=self.sample_tr_data, 
            val_instance=self.sample_val_data, 
            float_dtype=self._float_dtype)

    def se_kernel(self, querys, keys, values, reset=tf.constant(True, dtype=tf.bool)):
        return self._se_kernel(querys, keys, values, self.sigma, self.lengthscale, reset=reset)
        #return submodules.squared_exponential_kernel_fun(querys, keys, values, self.sigma, self.lengthscale)

    @snt.once
    def deep_se_kernel_init(self):
        self.se_kernel_all_init()
        self._deep_se_kernel = submodules.deep_se_kernel(
            embedding_layers=self._embedding_layers,
            kernel_dim=self.embedding_dim,
            initialiser=self.initialiser,
            nonlinearity=self._nonlinearity,
            train_instance=self.sample_tr_data,
            val_instance=self.sample_val_data,
            float_dtype=self._float_dtype
        )

    def deep_se_kernel(self, querys, keys, values, reset=tf.constant(True, dtype=tf.bool)):
        return self._deep_se_kernel(querys, keys, values, self.sigma, self.lengthscale, reset=reset)

    @snt.once
    def attention_block_init(self):
        config = {
            "rep": "mlp",
            "output_sizes": [self.embedding_dim] * self._embedding_layers,
            "att_type": self._attention_type,
            "normalise": tf.constant(True, tf.bool),
            "scale": 1.0,
            "l2_penalty_weight": self._l2_penalty_weight,
            "nonlinearity": self._nonlinearity
            }

        self.attention = submodules.Attention(config, train_instance=self.sample_tr_data, val_instance=self.sample_val_data)

    def attention_block(self, querys, keys, values, iter="", reset=tf.constant(True, dtype=tf.bool)):
        """dot-product kernel"""
        return self.attention(keys, querys, values, reset=reset)

    @snt.once
    def forward_decoder_with_decoder_init(self):
        self.decoder = submodules.decoder(
            embedding_dim=self.embedding_dim,
            classification=True,
            orthogonality_penalty_weight=self._orthogonality_penalty_weight, 
            initialiser=self.initialiser)

    def forward_decoder_with_decoder(self, cls_reprs):
        s = cls_reprs.shape.as_list()
        cls_reprs = tf.reshape(cls_reprs, s[:-1] + [self._num_classes, self._dim_reprs]) # split each representation into classes
        weights_dist_params, orthogonality_reg = self.decoder(cls_reprs) # get mean and variance of wn in LEO
        self._orthogonality_reg.assign(orthogonality_reg)
        stddev_offset = tf.math.sqrt(2. / (self.embedding_dim + self._num_classes)) #from LEO
        classifier_weights = self.sample(
            distribution_params=weights_dist_params,
            stddev_offset=stddev_offset)
        return classifier_weights

    def sample(self, distribution_params, stddev_offset=0.):
        """sample from a normal distribution"""
        return submodules.probabilistic_sample(
            distribution_params=distribution_params, 
            stddev_offset=stddev_offset,
            is_training=self.is_training)

    def predict_and_calculate_loss_and_acc(self, inputs, true_outputs, classifier_weights):
        """compute cross validation loss"""
        model_outputs = self.predict(inputs, classifier_weights) # return unnormalised probability of each class for each instance
        model_predictions = tf.math.argmax(
            input=model_outputs, axis=-1, output_type=self._int_dtype)
        accuracy = self.metric(model_predictions, tf.squeeze(true_outputs, axis=-1))
        self.metric.reset_state()
        return self.loss_fn(model_outputs, true_outputs), accuracy

    def _predict(self):
        """unnormalised class probabilities"""
        if self._no_decoder:
            return lambda inputs, weights: weights
        else:
            return self.predict_with_decoder

    def predict_with_decoder(self, inputs, weights):
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

    @property
    def _decoder_orthogonality_reg(self):
        return self._orthogonality_reg

    @property
    def additional_loss(self):
        return tf.constant(self._decoder_orthogonality_reg)



class MetaFunRegressor(MetaFunBase, snt.Module):
    def __init__(self, config, data_source="regression", no_batch=False, name="MetaFunRegressor"):
        """
        config: dict
            Configuation python dictionary, see ./config/sample.yaml
        data_source: str
            The sub-level name within the data level of config used for the problem
        name: str
            Name of classifier
        """
        super(MetaFunRegressor, self).__init__(config, no_batch, num_classes=1, name=name)

        # Decoder neural network size of last layers
        self._decoder_output_sizes = [40,40,2]
        self._loss_type = "mse"

    @snt.once
    def initialise(self, data_instance):
        """initialiser variables and functions"""
        self.sample_tr_data = data_instance.tr_input
        self.sample_val_data = data_instance.val_input
        self.embedding_dim = data_instance.tr_input.get_shape()[-1]

        self.forward_initialiser = self.forward_initialiser_base()
        self.forward_local_updater = self.forward_local_updater_base()
        self.forward_decoder = self.forward_decoder_base()
        self.forward_kernel_or_attention = self.forward_kernel_or_attention_base()
        
        # Extra internal functions
        self.predict_init()
        self.calculate_loss_and_metrics_init()

        # Change initialisation state
        self.has_initialised = True

        # Regularisation variables
        self.__call__(data_instance)
        self.regularise_variables = utils.get_linear_layer_variables(self)

    def __call__(self, data, is_training=tf.constant(True, dtype=tf.bool)):
        """
        data: dictionary-like form, with attributes "tr_input", "val_input", "tr_output", "val_output"
            Classification training/validation data of a task.
        is_training: bool
            If True, training mode
        """
        # Initialise Variables #TODO: is_training set as tf.constant in learner
        self.is_training.assign(is_training)
        self.reset.assign(True)
        assert self.has_initialised, "the learner is not initialised, please initialise via the method - .initialise"

        # Initialise r
        tr_reprs = self.forward_initialiser(data.tr_input, is_training=self.is_training)
        val_reprs = self.forward_initialiser(data.val_input, is_training=self.is_training)
        
        all_tr_reprs = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters)
        all_val_reprs = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters)

        for k in tf.range(self._num_iters):
            updates = self.forward_local_updater(r=tr_reprs, y=data.tr_output, x=data.tr_input)
            tr_updates, val_updates = tf.split(
                self.alpha * self.forward_kernel_or_attention(
                    querys=tf.concat([data.tr_input, data.val_input], axis=-2),
                    keys=data.tr_input,
                    values=updates,
                    reset=self.reset),
                num_or_size_splits=[data.tr_input.shape[-2], data.val_input.shape[-2]],
                axis=-2
                )
            tr_reprs += tr_updates
            val_reprs += val_updates
            all_tr_reprs = all_tr_reprs.write(k, tr_reprs)
            all_val_reprs = all_val_reprs.write(k, val_reprs)

            self.reset.assign(False, dtype=tf.bool)
        return tf.constant(2.), all_tr_reprs.stack(), all_val_reprs.stack()

    @snt.once
    def initialiser_all_init(self):
        self.alpha =  tf.Variable( # Inner learning rate
            initial_value=tf.constant_initializer(self._initial_inner_lr)(
                shape=[1,1],
                dtype=self._float_dtype
            ),
            trainable=True,
            name="alpha"
            )
        
    @snt.once
    def constant_initialiser_init(self):
        self.initialiser_all_init()
        if self._initial_state_type == tf.constant("zero", dtype=tf.string):
            self._constant_initialiser = submodules.constant_initialiser(
                    dim_reprs=self._dim_reprs, 
                    float_dtype=self._float_dtype, 
                    classification=False, 
                    no_batch=self._no_batch, 
                    trainable=False)
        else:
            self._constant_initialiser = submodules.constant_initialiser(
                    dim_reprs=self._dim_reprs, 
                    float_dtype=self._float_dtype, 
                    classification=False, 
                    no_batch=self._no_batch, 
                    trainable=True)

    def constant_initialiser(self, x, is_training=True):
        return self._constant_initialiser(x)

    @snt.once
    def parametric_initialiser_init(self):
        self.initialiser_all_init()
        self._parametric_initialiser = submodules.parametric_initialiser(
                        nn_size=self._nn_size,
                        nn_layers=self._nn_layers,
                        dim_reprs=self._dim_reprs,
                        classification=False,
                        dropout_rate=self._dropout_rate,
                        initialiser=self.initialiser,
                        nonlinearity=self._nonlinearity
                        )

    def parametric_initialiser(self, x, is_training):
        return self._parametric_initialiser(x, is_training)

    @snt.once
    def gradient_local_updater_init(self):
        # Inner learning rate for each functional representation for each class
        self.lr = tf.Variable(
            initial_value=tf.constant_initializer(1.0)(
                shape=[1, self._dim_reprs],
                dtype=self._float_dtype,
            ),
            trainable=True,
            name="lr"
        )

    def gradient_local_updater(self, r, y, x=None, iter=""):
        """functional gradient update instead of neural update"""
        with tf.GradientTape() as tape:
            tape.watch(r)
            weights = self.forward_decoder(r)
            tr_mu, tr_sigma = self.predict(x, weights)
            tr_loss, tr_mse = self.calculate_loss_and_metrics(target_y=y, mus=tr_mu, sigmas=tr_sigma)

            batch_tr_loss = tf.reduce_mean(tr_loss)
        loss_grad = tape.gradient(batch_tr_loss, r)
        updates = - self.lr * loss_grad
        return updates

    def neural_local_updater_init(self):
        self.neural_local_updater = submodules.neural_local_updater(
            nn_size=self._nn_size, 
            nn_layers=self._nn_layers, 
            dim_reprs=self._dim_reprs, 
            classification=False,
            initialiser=self.initialiser, 
            nonlinearity=self._nonlinearity,
            no_batch=self._no_batch,
            xNone=self._neural_updater_concat_x)

    def neural_local_updater(self, r, y, x, iter=""):
        return self._neural_local_updater(r=r, y=y, x=x, iter=iter)

    @snt.once
    def se_kernel_all_init(self):
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

    @snt.once
    def se_kernel_init(self):
        self.se_kernel_all_init()
        self._se_kernel = submodules.squared_exponential_kernel(
            train_instance=self.sample_tr_data, 
            val_instance=self.sample_val_data, 
            float_dtype=self._float_dtype)
    
    def se_kernel(self, querys, keys, values, reset=tf.constant(True, dtype=tf.bool)):
        return self._se_kernel(querys, keys, values, self.sigma, self.lengthscale, reset=reset)

    @snt.once
    def deep_se_kernel_init(self):
        self.se_kernel_all_init()
        self._deep_se_kernel = submodules.deep_se_kernel(
            embedding_layers=self._embedding_layers,
            kernel_dim=self._nn_size,
            initialiser=self.initialiser,
            nonlinearity=self._nonlinearity,
            train_instance=self.sample_tr_data,
            val_instance=self.sample_val_data,
            float_dtype=self._float_dtype
        )

    def deep_se_kernel(self, querys, keys, values, reset=tf.constant(True, dtype=tf.bool)):
        return self._deep_se_kernel(querys, keys, values, self.sigma, self.lengthscale, reset=reset)


    @snt.once
    def attention_block_init(self):
        config = {
            "rep": "mlp",
            "output_sizes": [self._nn_size] * self._embedding_layers,
            "att_type": self._attention_type,
            "normalise": tf.constant(True, tf.bool),
            "scale": 1.0,
            "l2_penalty_weight": self._l2_penalty_weight,
            "nonlinearity": self._nonlinearity
            }

        self.attention = submodules.Attention(config, train_instance=self.sample_tr_data, val_instance=self.sample_val_data)

    def attention_block(self, querys, keys, values, iter="", reset=tf.constant(True, dtype=tf.bool)):
        """dot-product kernel"""
        return self.attention(keys, querys, values, reset=reset)

    def forward_decoder_with_decoder_init(self):
        self.decoder = submodules.decoder(
            embedding_dim=self.embedding_dim,
            initialiser=self.initialiser,
            classification=False,
            nn_size=self._nn_size,
            nn_layers=self._nn_layers,
            nonlinearity=self._nonlinearity,
            repr_as_inputs=self._repr_as_inputs,
            regression_output_sizes=self._decoder_output_sizes
        )

    def forward_decoder_with_decoder(self, reprs):
        weights_dist_params = self.decoder(reprs)
        stddev_offset = tf.math.sqrt(2. / self._nn_size)
        weights = self.sample(weights_dist_params, stddev_offset)
        return weights

    def sample(self, distribution_params, stddev_offset=0.):
        """a deterministic sample"""
        return submodules.deterministic_sample(
            distribution_params=distribution_params, 
            stddev_offsets=stddev_offset)

    @snt.once
    def calculate_loss_and_metrics_init(self):

        def mse_loss(target_y, mus, sigmas, coeffs=None):
            mu, sigma = mus, sigmas
            mse = self.loss_fn(mu, target_y)
            return mse, mse

        def log_prob_loss(target_y, mus, sigmas, coeffs=None):
            mu, sigma = mus, sigmas
            dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
            loss = -dist.log_prob(target_y)
            mse = self.loss_fn(mu, target_y)
            return loss, mse

        if self._loss_type == "mse":
            self._calculate_loss_and_metrics =  mse_loss
        elif self._loss_type == "log_prob":
            self._log_prob_loss =  log_prob_loss
        else:
            raise NameError("unknown loss type")

    def calculate_loss_and_metrics(self, target_y, mus, sigmas, coeffs=None):
        return self._calculate_loss_and_metrics(target_y=target_y, mus=mus, sigmas=sigmas, coeffs=coeffs)


    @snt.once
    def predict_init(self):
        """ backend of decoder to produce mean and variance of predictions"""
        def predict_no_decoder1(inputs, weights):
            return weights, tf.ones_like(weights) * 0.5

        def predict_no_decoder2(inputs, weights):
            return tf.split(weights, 2, axis=-1)

        def predict_repr_as_inputs(inputs, weights):
            preds = self._predict_repr_as_inputs(inputs=inputs, weights=weights)
            return _split(preds)

        def predict_not_repr_as_inputs(inputs, weights):
            preds = self.custom_MLP(inputs=inputs, weights=weights)
            return _split(preds)
        
        def _split(preds):
            mu, log_sigma = tf.split(preds, 2, axis=-1)
            sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)
            return mu, sigma

        if self._no_decoder:
            if self._dim_reprs == 1:
                self._predict = predict_no_decoder1
            elif self._dim_reprs == 2:
                self._predict = predict_no_decoder2
            else:
                raise Exception("num_reprs must <=2 if no_decoder")
        if self._repr_as_inputs:
            self._predict_repr_as_inputs = submodules.predict_repr_as_inputs(
                output_sizes=self._decoder_output_sizes,
                initialiser=self.initialiser,
                nonlinearity=self._nonlinearity
            )
            self._predict =  predict_repr_as_inputs
        else:
            self.custom_MLP = submodules.custom_MLP(
                output_sizes=self._decoder_output_sizes,
                embedding_dim=self.embedding_dim,
                nonlinearity=self._nonlinearity)
            self._predict =  predict_not_repr_as_inputs

    def predict(self, inputs, weights):
        return self._predict(inputs=inputs, weights=weights)

    def loss_fn(self, model_outputs, labels):
        return tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE)(
                y_true=labels, y_pred=model_outputs)

    @property
    def additional_loss(self):
        return tf.constant(0., dtype=self._float_dtype)







    

if __name__ == "__main__":
    from utils import parse_config
    import os
    import numpy as np
    import collections
    config = parse_config(os.path.join(os.path.dirname(__file__),"config/debug.yaml"))
    module = MetaFunClassifier(config=config)
    tf.random.set_seed(1234)
    np.random.seed(1234)

    # ClassificationDescription = collections.namedtuple(
    # "ClassificationDescription",
    # ["tr_input", "tr_output", "val_input", "val_output"])
    
    # data = ClassificationDescription(
    # tf.constant(np.random.random([10,10]),dtype=tf.float32),
    # tf.constant(np.random.uniform(1,10,10).reshape(-1,1),dtype=tf.int32),
    # tf.constant(np.random.random([10,10]),dtype=tf.float32),
    # tf.constant(np.random.uniform(1,10,10).reshape(-1,1),dtype=tf.int32))

    # print(module(data))
    print("Classification")
    from data.leo_imagenet import DataProvider
    dataloader = DataProvider("trial", config)
    dat = dataloader.generate()
    for i in dat.take(1):
        module.initialise(i)

    @tf.function
    def trial(x, is_training=tf.constant(True,tf.bool)):
        l,_,_ = module(x, is_training=is_training)
        return l
    print("DEBUGGGGGGGGGGGGGGG")
    for i in dat.take(1):
        print(trial(i, is_training=False))
        print(module(i, is_training=False)[0])

    # print("Regression")
    # module2 = MetaFunRegressor(config=config)
    # ClassificationDescription = collections.namedtuple(
    # "ClassificationDescription",
    # ["tr_input", "tr_output", "val_input", "val_output"])
    
    # data_reg = ClassificationDescription(
    # tf.constant(np.random.random([2, 10,10]),dtype=tf.float32),
    # tf.constant(np.random.random([2,10,1]),dtype=tf.float32),
    # tf.constant(np.random.random([2, 10,10]),dtype=tf.float32),
    # tf.constant(np.random.random([2,10,1]),dtype=tf.float32))

    # module2.initialise(data_reg)
    # @tf.function
    # def trial(x):
    #     l,_,_ = module2(x)
    #     return l

    # print("DEBUGGGGGGGGGGGGGGG")
    # print(trial(data_reg))

