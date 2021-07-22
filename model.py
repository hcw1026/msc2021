from functools import partial
import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt
import submodules as submodules
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
        self._deterministic_decoder = tf.constant(_config["deterministic_decoder"], dtype=tf.bool)
        self._kernel_sigma_init = _config["kernel_sigma_init"]
        self._kernel_lengthscale_init = _config["kernel_lengthscale_init"]
        self._indp_iter = tf.constant(_config["indp_iter"], dtype=tf.bool)

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
        self._orthogonality_reg = tf.constant(0., dtype=self._float_dtype)
        self.is_training = tf.Variable(True, dtype=tf.bool, trainable=False)
        self.embedding_dim = tf.constant([1])
        self.has_initialised = False

        # Classification or Regression
        self._num_classes = num_classes
        self._repr_as_inputs = config["Model"]["comp"]["repr_as_inputs"]
        self._neural_updater_concat_x = config["Model"]["comp"]["neural_updater_concat_x"]

        # Define metric
        self._metric_names = []

        if self._no_decoder:
            self._dim_reprs = 1

    @snt.once
    def additional_initialise(self, data_instance): # for extra initialisation steps
        pass

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
                return (self.se_kernel_precompute, self.se_kernel_backend)
            elif self._kernel_type == tf.constant("deep_se", dtype=tf.string):
                self.deep_se_kernel_init()
                return (self.deep_se_kernel_precompute, self.deep_se_kernel_backend)
            else:
                raise NameError("Unknown kernel type")
        else:
            self.attention_block_init()
            return (self.attention_block_precompute, self.attention_block_backend)

    def forward_decoder_base(self):
        """decode and sample weight for final outpyt layer"""
        if self._no_decoder: 
            # use functional representation directly as the predictor, used in ablation study
            return self.forward_decoder_without_decoder
        else:
            self.forward_decoder_with_decoder_init()
            return self.forward_decoder_with_decoder

    @property
    def get_metric_name(self):
        return self._metric_names

    @property
    def get_regularise_variables(self):
        return self.regularise_variables

    @property
    def additional_loss(self):
        return tf.constant(0., dtype=self._float_dtype)


class MetaFunClassifier(MetaFunBase):
    def __init__(self, config, data_source="leo_imagenet", no_batch=False, name="MetaFunClassifier", **kwargs): #TODO: remove no_batch
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
        self._metric_names = ["acc"] # must be in the order of metric output

        # Metric initialisation
        #self.metric = tf.keras.metrics.BinaryAccuracy(name="inner accuracy", dtype=self._float_dtype)

    @snt.once
    def initialise(self, data_instance):
        """initialiser variables and functions"""

        self.embedding_dim = data_instance.tr_input.get_shape()[-1]

        self.forward_initialiser = self.forward_initialiser_base()
        self.forward_local_updater = self.forward_local_updater_base()
        self.forward_decoder = self.forward_decoder_base()
        self.forward_kernel_or_attention_precompute, self.forward_kernel_or_attention = self.forward_kernel_or_attention_base()

        # Extra internal functions
        self.predict_init()
        self.sample_init()

        # Change initialisation state
        self.additional_initialise(data_instance=data_instance)
        self.has_initialised = True
        #self.__call__(data_instance)

        # Regularisation variables
        self.regularise_variables = utils.get_linear_layer_variables(self)

    @snt.once
    def additional_initialise(self, data_instance):
        """for data-specific initialisation and any extra initialisation"""

        self.sample_tr_data = data_instance.tr_input
        self.sample_val_data = data_instance.val_input

        # Deduce precompute shape of forward_kernel_or_attention_precompute
        self.precomputed_init = tf.zeros_like(self.forward_kernel_or_attention_precompute(
            querys=tf.concat([self.sample_tr_data, self.sample_val_data], axis=-2),
            keys=self.sample_tr_data,
            values=None,
            recompute=True,
            precomputed=None,
            iteration=0
        ))

    def __call__(self, data, is_training=tf.constant(True, dtype=tf.bool)):
        """
        data: dictionary-like form, with attributes "tr_input", "val_input", "tr_output", "val_output"
            Classification training/validation data of a task.
        is_training: bool
            If True, training mode
        """
        # Initialise Variables #TODO: is_training set as tf.constant in learner
        self.is_training.assign(is_training)
        assert self.has_initialised, "the learner is not initialised, please initialise via the method - .initialise"

        # Initialise r
        tr_reprs = self.forward_initialiser(data.tr_input, is_training=self.is_training)
        val_reprs = self.forward_initialiser(data.val_input, is_training=self.is_training)

        # Precompute target context interaction for kernel/attention
        precomputed = self.forward_kernel_or_attention_precompute(
            querys=tf.concat([data.tr_input, data.val_input],axis=-2), 
            keys=data.tr_input,
            recompute=tf.math.logical_not(self._indp_iter), # if not indepnedent iteration, precompute 
            precomputed=self.precomputed_init,
            values=None,
            iteration=0)

        # Iterative functional updating
        for k in range(self._num_iters):
            updates = self.forward_local_updater(tr_reprs, data.tr_output, data.tr_input, iteration=k) #return negative u

            precomputed = self.forward_kernel_or_attention_precompute(
                querys=tf.concat([data.tr_input, data.val_input],axis=-2), 
                keys=data.tr_input,
                recompute=self._indp_iter,
                precomputed=precomputed,
                values=None,
                iteration=k)

            tr_updates, val_updates = tf.split(
                self.alpha * self.forward_kernel_or_attention(
                    querys=None, 
                    keys=None, 
                    precomputed=precomputed,
                    values=updates), 
                num_or_size_splits=[tf.shape(data.tr_input)[-2], tf.shape(data.val_input)[-2]], 
                axis=-2)
            tr_reprs += tr_updates
            val_reprs += val_updates

        # Decode functional representation and compute loss and metric
        classifier_weights, tr_orth = self.forward_decoder(tr_reprs)
        tr_loss, batch_tr_metric = self.predict_and_calculate_loss_and_acc(data.tr_input, data.tr_output, classifier_weights)
        classifier_weights, val_orth = self.forward_decoder(val_reprs)
        val_loss, batch_val_metric = self.predict_and_calculate_loss_and_acc(data.val_input, data.val_output, classifier_weights)


        # # Aggregate loss in a batch
        # batch_tr_loss = tf.math.reduce_mean(tr_loss)
        # batch_val_loss = tf.math.reduce_mean(val_loss)

        #Additional regularisation penalty
        #return batch_val_loss + self._decoder_orthogonality_reg, batch_tr_metric, batch_val_metric  #TODO:? need weights for l2

        return val_loss, tr_orth, batch_tr_metric, batch_val_metric

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

    def gradient_local_updater(self, r, y, x=None, iteration=0):
        """functional gradient update instead of neural update"""
        with tf.GradientTape() as tape:
            tape.watch(r) #watch r
            classifier_weights, _ = self.forward_decoder(r) # sample w_n from LEO
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
            no_batch=self._no_batch,
            num_iters=self._num_iters,
            indp_iter=self._indp_iter)

    def neural_local_updater(self, r, y, x, iteration=0):
        return self._neural_local_updater(r=r, y=y, x=x, iteration=iteration)

    @snt.once
    def se_kernel_all_init(self):
        # Kernel sigma
        self.sigma = tf.Variable(
            initial_value=tf.constant_initializer(self._kernel_sigma_init)(
                shape=(),
                dtype=self._float_dtype
            ),
            trainable=True,
            name="kernel_sigma"
        )

        # Kernel lengthscale
        self.lengthscale = tf.Variable(
            initial_value=tf.constant_initializer(self._kernel_lengthscale_init)(
                shape=(),
                dtype=self._float_dtype
            ),
            trainable=True,
            name="kernel_lengthscale"
        )

    @snt.once
    def se_kernel_init(self):
        self.se_kernel_all_init()
        self._se_kernel = submodules.squared_exponential_kernel(complete_return=False)

    def se_kernel_precompute(self, querys, keys, recompute, precomputed, values=None, iteration=0):
        return self._se_kernel(querys=querys, keys=keys, recompute=recompute, precomputed=precomputed, sigma=self.sigma, lengthscale=self.lengthscale)

    def se_kernel_backend(self, querys, keys, precomputed, values): # use this input format for generality
        return self._se_kernel.backend(query_key=precomputed, values=values)

    @snt.once
    def deep_se_kernel_init(self):
        self.se_kernel_all_init()
        self._deep_se_kernel = submodules.deep_se_kernel(
            embedding_layers=self._embedding_layers,
            kernel_dim=self.embedding_dim,
            initialiser=self.initialiser,
            nonlinearity=self._nonlinearity,
            complete_return=False,
            num_iters=self._num_iters, 
            indp_iter=self._indp_iter
        )

    def deep_se_kernel_precompute(self, querys, keys, recompute, precomputed, values=None, iteration=0):
        return self._deep_se_kernel(querys=querys, keys=keys, recompute=recompute, precomputed=precomputed, sigma=self.sigma, lengthscale=self.lengthscale, iteration=iteration)

    def deep_se_kernel_backend(self, querys, keys, precomputed, values):
        return self._deep_se_kernel.backend(query_key=precomputed, values=values)

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

        self.attention = submodules.Attention(config=config, num_iters=self._num_iters, indp_iter=self._indp_iter, complete_return=False)

    def attention_block_precompute(self, querys, keys, recompute, precomputed, values=None, iteration=0):
        """dot-product kernel"""
        return self.attention(keys=keys, querys=querys, recompute=recompute, precomputed=precomputed, values=values, iteration=iteration)

    def attention_block_backend(self, querys, keys, precomputed, values):
        return self.attention.backend(weights=precomputed, values=values)

    @snt.once
    def forward_decoder_with_decoder_init(self):
        self.decoder = submodules.decoder(
            embedding_dim=self.embedding_dim,
            classification=True,
            orthogonality_penalty_weight=self._orthogonality_penalty_weight, 
            initialiser=self.initialiser)

    def forward_decoder_with_decoder(self, cls_reprs):
        s = tf.shape(cls_reprs)
        cls_reprs = tf.reshape(cls_reprs, tf.concat([s[:-1], tf.constant([self._num_classes, self._dim_reprs], dtype=tf.int32)], axis=0))
        weights_dist_params, orthogonality_reg = self.decoder(cls_reprs) # get mean and variance of wn in LEO
        stddev_offset = tf.math.sqrt(2. / (self.embedding_dim + self._num_classes)) #from LEO
        classifier_weights = self.sample(
            distribution_params=weights_dist_params,
            stddev_offset=stddev_offset)
        return classifier_weights, orthogonality_reg

    def forward_decoder_without_decoder(self, cls_reprs):
        return cls_reprs, tf.constant(0.)

    @snt.once
    def sample_init(self):
        if self._deterministic_decoder:
            self._sample = lambda distribution_params, stddev_offset, is_training: submodules.deterministic_sample(distribution_params=distribution_params, stddev_offset=stddev_offset)
        else:
            self._sample = lambda distribution_params, stddev_offset, is_training: submodules.probabilistic_sample(distribution_params=distribution_params, stddev_offset=stddev_offset, is_training=is_training)

    def sample(self, distribution_params, stddev_offset=0.):
        return self._sample(
            distribution_params=distribution_params, 
            stddev_offset=stddev_offset,
            is_training=self.is_training)

    def predict_and_calculate_loss_and_acc(self, inputs, true_outputs, classifier_weights):
        """compute cross validation loss"""
        model_outputs = self.predict(inputs, classifier_weights) # return unnormalised probability of each class for each instance
        model_predictions = tf.math.argmax(
            input=model_outputs, axis=-1, output_type=self._int_dtype)
        # accuracy = self.metric(model_predictions, tf.squeeze(true_outputs, axis=-1))
        # self.metric.reset_state()
        accuracy = tf.cast(tf.equal(model_predictions, tf.squeeze(true_outputs, axis=-1)), dtype=self._float_dtype)
        return self.loss_fn(model_outputs, true_outputs), [accuracy]

    def predict_init(self):
        """unnormalised class probabilities"""

        def predict_with_decoder(inputs, weights):
            after_dropout = self.dropout(inputs, training=self.is_training)
            preds = tf.linalg.matvec(weights, after_dropout) # (x^Tw)_k - i=instance, m=class, k=features
            return preds

        if self._no_decoder:
            self._predict = lambda inputs, weights: weights
        else:
            self._predict = predict_with_decoder

    def predict(self, inputs, weights):
        return self._predict(inputs=inputs, weights=weights)

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


class MetaFunRegressor(MetaFunBase):
    def __init__(self, config, data_source="regression", no_batch=False, name="MetaFunRegressor", **kwargs):
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
        self._decoder_output_sizes = [self._nn_size] * (self._nn_layers-1) + [2]
        self._loss_type = config["Train"]["loss_type"]

        self._metric_names = ["mse", "logprob"] # must be in the order of metric output

    @snt.once
    def initialise(self, data_instance):
        """initialiser variables and functions"""

        self.embedding_dim = data_instance.tr_input.get_shape()[-1]

        self.forward_initialiser = self.forward_initialiser_base()
        self.forward_local_updater = self.forward_local_updater_base()
        self.forward_decoder = self.forward_decoder_base()
        self.forward_kernel_or_attention_precompute, self.forward_kernel_or_attention = self.forward_kernel_or_attention_base()

        # Extra internal functions
        self.predict_init()
        self.calculate_loss_and_metrics_init()
        self.sample_init()

        # Change initialisation state
        self.additional_initialise(data_instance=data_instance)
        self.has_initialised = True
        #self.__call__(data_instance)

        # Regularisation variables
        self.regularise_variables = utils.get_linear_layer_variables(self)

    @snt.once
    def additional_initialise(self, data_instance):
        """for data-specific initialisation and any extra initialisation"""

        self.sample_tr_data = data_instance.tr_input
        self.sample_val_data = data_instance.val_input

        # Deduce precompute shape of forward_kernel_or_attention_precompute
        self.precomputed_init = tf.zeros_like(self.forward_kernel_or_attention_precompute(
            querys=tf.concat([self.sample_tr_data, self.sample_val_data], axis=-2),
            keys=self.sample_tr_data,
            values=None,
            recompute=True,
            precomputed=None,
            iteration=0
        ))

    def __call__(self, data, is_training=tf.constant(True, dtype=tf.bool)):
        """
        data: dictionary-like form, with attributes "tr_input", "val_input", "tr_output", "val_output"
            Classification training/validation data of a task.
        is_training: bool
            If True, training mode
        """
        # Initialise Variables #TODO: is_training set as tf.constant in learner
        self.is_training.assign(is_training)
        assert self.has_initialised, "the learner is not initialised, please initialise via the method - .initialise"

        all_tr_reprs = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters+1)
        all_val_reprs = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters+1)

        # Initialise r
        tr_reprs = self.forward_initialiser(data.tr_input, is_training=self.is_training)
        val_reprs = self.forward_initialiser(data.val_input, is_training=self.is_training)
        
        all_tr_reprs = all_tr_reprs.write(0, tr_reprs)
        all_val_reprs = all_val_reprs.write(0, val_reprs)

        # Precompute target context interaction for kernel/attention
        precomputed = self.forward_kernel_or_attention_precompute(
            querys=tf.concat([data.tr_input, data.val_input],axis=-2), 
            keys=data.tr_input,
            recompute=tf.math.logical_not(self._indp_iter),
            precomputed=self.precomputed_init,
            values=None,
            iteration=0)

        # Iterative functional updating
        for k in range(self._num_iters):
            updates = self.forward_local_updater(r=tr_reprs, y=data.tr_output, x=data.tr_input, iteration=k)

            precomputed = self.forward_kernel_or_attention_precompute(
                querys=tf.concat([data.tr_input, data.val_input],axis=-2), 
                keys=data.tr_input,
                recompute=self._indp_iter,
                precomputed=precomputed,
                values=None,
                iteration=k)

            tr_updates, val_updates = tf.split(
                self.alpha * self.forward_kernel_or_attention(
                    querys=None,
                    keys=None,
                    precomputed=precomputed,
                    values=updates),
                num_or_size_splits=[tf.shape(data.tr_input)[-2], tf.shape(data.val_input)[-2]],
                axis=-2
                )
            tr_reprs += tr_updates
            val_reprs += val_updates
            all_tr_reprs = all_tr_reprs.write(1+k, tr_reprs)
            all_val_reprs = all_val_reprs.write(1+k, val_reprs)

        # Decode functional representation and compute loss and metric
        all_val_mu = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters+1)
        all_val_sigma = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters+1)
        all_tr_mu = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters+1)
        all_tr_sigma = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters+1)

        for k in range(self._num_iters+1): # store intermediate predictions
            weights = self.forward_decoder(all_tr_reprs.read(k))
            tr_mu, tr_sigma = self.predict(inputs=data.tr_input, weights=weights)
            weights = self.forward_decoder(all_val_reprs.read(k))
            val_mu, val_sigma = self.predict(inputs=data.val_input, weights=weights)
            all_val_mu = all_val_mu.write(k, val_mu)
            all_val_sigma = all_val_sigma.write(k, val_sigma)
            all_tr_mu = all_tr_mu.write(k, tr_mu)
            all_tr_sigma = all_tr_sigma.write(k, tr_sigma)

        tr_loss, tr_metric = self.calculate_loss_and_metrics(
            target_y=data.tr_output,
            mus=tr_mu,
            sigmas=tr_sigma)
        val_loss, val_metric = self.calculate_loss_and_metrics(
            target_y=data.val_output,
            mus=val_mu,
            sigmas=val_sigma
        )

        all_val_mu = all_val_mu.stack()
        all_val_sigma = all_val_sigma.stack()
        all_tr_mu = all_tr_mu.stack()
        all_tr_sigma = all_tr_sigma.stack()

        additional_loss = tf.constant(0., dtype=self._float_dtype)

        return val_loss, additional_loss, tr_metric, val_metric, all_val_mu, all_val_sigma, all_tr_mu, all_tr_sigma

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

    def gradient_local_updater(self, r, y, x=None, iteration=0):
        """functional gradient update instead of neural update"""
        with tf.GradientTape() as tape:
            tape.watch(r)
            weights = self.forward_decoder(r)
            tr_mu, tr_sigma = self.predict(x, weights)
            tr_loss, _ = self.calculate_loss_and_metrics(target_y=y, mus=tr_mu, sigmas=tr_sigma)

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
            xNone=self._neural_updater_concat_x,
            num_iters=self._num_iters,
            indp_iter=self._indp_iter)

    def neural_local_updater(self, r, y, x, iteration=0):
        return self._neural_local_updater(r=r, y=y, x=x, iteration=iteration)

    @snt.once
    def se_kernel_all_init(self):
        # Kernel sigma
        self.sigma = tf.Variable(
            initial_value=tf.constant_initializer(self._kernel_sigma_init)(
                shape=(),
                dtype=self._float_dtype
            ),
            trainable=True,
            name="kernel_sigma"
        )

        # Kernel lengthscale
        self.lengthscale = tf.Variable(
            initial_value=tf.constant_initializer(self._kernel_lengthscale_init)(
                shape=(),
                dtype=self._float_dtype
            ),
            trainable=True,
            name="kernel_lengthscale"
        )

    @snt.once
    def se_kernel_init(self):
        self.se_kernel_all_init()
        self._se_kernel = submodules.squared_exponential_kernel(complete_return=False)
    
    def se_kernel_precompute(self, querys, keys, recompute, precomputed, values=None, iteration=0):
        return self._se_kernel(querys=querys, keys=keys, recompute=recompute, precomputed=precomputed, sigma=self.sigma, lengthscale=self.lengthscale, iteration=iteration)

    def se_kernel_backend(self, querys, keys, precomputed, values):
        return self._se_kernel.backend(query_key=precomputed, values=values)

    @snt.once
    def deep_se_kernel_init(self):
        self.se_kernel_all_init()
        self._deep_se_kernel = submodules.deep_se_kernel(
            embedding_layers=self._embedding_layers,
            kernel_dim=self._nn_size,
            initialiser=self.initialiser,
            nonlinearity=self._nonlinearity,
            complete_return=False,
            num_iters=self._num_iters, 
            indp_iter=self._indp_iter
        )

    def deep_se_kernel_precompute(self, querys, keys, recompute, precomputed, values=None, iteration=0):
        return self._deep_se_kernel(querys=querys, keys=keys, recompute=recompute, precomputed=precomputed, sigma=self.sigma, lengthscale=self.lengthscale, iteration=iteration)

    def deep_se_kernel_backend(self, querys, keys, precomputed, values):
        return self._deep_se_kernel.backend(query_key=precomputed, values=values)

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

        self.attention = submodules.Attention(config=config, num_iters=self._num_iters, indp_iter=self._indp_iter, complete_return=False)

    def attention_block_precompute(self, querys, keys, recompute, precomputed, values=None, iteration=0):
        """dot-product kernel"""
        return self.attention(keys=keys, querys=querys, recompute=recompute, precomputed=precomputed, values=values, iteration=iteration)

    def attention_block_backend(self, querys, keys, precomputed, values):
        return self.attention.backend(weights=precomputed, values=values)

    @snt.once
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
        stddev_offset = tf.math.sqrt(1. / self._nn_size)
        weights = self.sample(weights_dist_params, stddev_offset)
        return weights

    def forward_decoder_without_decoder(self, reprs):
        return reprs

    @snt.once
    def sample_init(self):
        if self._deterministic_decoder:
            self._sample = lambda distribution_params, stddev_offset, is_training: submodules.deterministic_sample(distribution_params=distribution_params, stddev_offset=stddev_offset)
        else:
            self._sample = lambda distribution_params, stddev_offset, is_training: submodules.probabilistic_sample(distribution_params=distribution_params, stddev_offset=stddev_offset, is_training=is_training)

    def sample(self, distribution_params, stddev_offset=0.):
        return self._sample(
            distribution_params=distribution_params, 
            stddev_offset=stddev_offset,
            is_training=self.is_training)

    @snt.once
    def calculate_loss_and_metrics_init(self):

        def mse_loss(target_y, mus, sigmas, coeffs=None):
            mu, sigma = mus, sigmas
            mse = self.loss_fn(mu, target_y)
            return mse


        def log_prob_loss(target_y, mus, sigmas, coeffs=None):
            mu, sigma = mus, sigmas
            dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
            loss = - dist.log_prob(target_y)
            return loss

        def loss_and_metric(loss_fn, target_y, mus, sigmas, coeffs=None):
            loss = loss_fn(target_y=target_y, mus=mus, sigmas=sigmas, coeffs=coeffs)
            mse = mse_loss(target_y=target_y, mus=mus, sigmas=sigmas, coeffs=coeffs)
            if tf.shape(target_y)[-2] != tf.constant(0, dtype=tf.int32): # to avoid nan when computing metrics
                logprob = - tf.math.reduce_mean(log_prob_loss(target_y=target_y, mus=mus, sigmas=sigmas, coeffs=coeffs), axis=-1, keepdims=True)
            else:
                logprob = target_y[...,0] #empty shape
            return loss, [mse, logprob]

        if self._loss_type == "mse":
            self._calculate_loss_and_metrics =  partial(loss_and_metric, loss_fn=mse_loss)
        elif self._loss_type == "logprob":
            self._calculate_loss_and_metrics =  partial(loss_and_metric, loss_fn=log_prob_loss)
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
        elif self._repr_as_inputs:
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
    

class MetaFunBaseV2(MetaFunBase):
    def __init__(self, config, no_batch, num_classes=1, name="MetaFunV2", **kwargs):
        super(MetaFunBaseV2, self).__init__(config=config, no_batch=no_batch, num_classes=num_classes, name=name)
        _config = config["Model"]["ff"]
        self._use_ff = _config["use_ff"]
        self._ff_stddev_init = _config["stddev_init"]
        self._ff_num_freq = _config["num_freq"]
        self._ff_learnable = _config["learnable"]

    @snt.once
    def additional_initialise(self, data_instance):
        if self._use_ff:
            self.fourier_features_init()
            self.fourier_features = self.fourier_features_compute
        else:
            self.fourier_features = lambda X, recompute, precomputed, iteration: X

        self.sample_tr_data = data_instance.tr_input
        self.sample_val_data = data_instance.val_input

        self.tr_data_ff_init = self.fourier_features(X=self.sample_tr_data, recompute=True, precomputed=None, iteration=0)
        self.val_data_ff_init = self.fourier_features(X=self.sample_val_data, recompute=True, precomputed=None, iteration=0)

        # Deduce precompute shape of forward_kernel_or_attention_precompute
        self.precomputed_init = tf.zeros_like(self.forward_kernel_or_attention_precompute(
            querys=tf.concat([self.tr_data_ff_init, self.val_data_ff_init], axis=-2),
            keys=self.tr_data_ff_init,
            values=None,
            recompute=True,
            precomputed=None,
            iteration=0
        ))

    def fourier_features_init(self):
        self._fourier_features = submodules.FourierFeatures(
            stddev_init=self._ff_stddev_init, 
            embedding_dim=self.embedding_dim, 
            num_freq=self._ff_num_freq, 
            learnable=self._ff_learnable,
            num_iters=self._num_iters, 
            indp_iter=self._indp_iter, 
            float_dtype=self._float_dtype)
        self.embedding_dim = self._fourier_features.embedding_dim
    
    def fourier_features_compute(self, X, recompute, precomputed, iteration=0):
        return self._fourier_features(X=X, recompute=recompute, precomputed=precomputed, iteration=iteration)


class MetaFunClassifierV2(MetaFunBaseV2, MetaFunClassifier):
    def __init__(self, config, data_source="leo_imagenet", no_batch=False, name="MetaFunClassifierV2"):
        self._num_classes = config["Data"][data_source]["num_classes"]
        super(MetaFunClassifierV2, self).__init__(config=config, data_source=data_source, num_classes=self._num_classes, no_batch=no_batch, name=name)

    def __call__(self, data, is_training=tf.constant(True, dtype=tf.bool)):
        """
        data: dictionary-like form, with attributes "tr_input", "val_input", "tr_output", "val_output"
            Classification training/validation data of a task.
        is_training: bool
            If True, training mode
        """
        # Initialise Variables #TODO: is_training set as tf.constant in learner
        self.is_training.assign(is_training)
        assert self.has_initialised, "the learner is not initialised, please initialise via the method - .initialise"

        # Initialise r
        tr_reprs = self.forward_initialiser(data.tr_input, is_training=self.is_training)
        val_reprs = self.forward_initialiser(data.val_input, is_training=self.is_training)

        # Fourier features
        tr_input_ff = self.fourier_features(
            X=data.tr_input, 
            recompute=tf.math.logical_not(self._indp_iter), 
            precomputed=self.tr_data_ff_init,
            iteration=0)

        val_input_ff = self.fourier_features(
            X=data.val_input, 
            recompute=tf.math.logical_not(self._indp_iter), 
            precomputed=self.val_data_ff_init,
            iteration=0)

        # Precompute target context interaction for kernel/attention
        precomputed = self.forward_kernel_or_attention_precompute(
            querys=tf.concat([tr_input_ff, val_input_ff],axis=-2), 
            keys=tr_input_ff,
            recompute=tf.math.logical_not(self._indp_iter), # if not indepnedent iteration, precompute 
            precomputed=self.precomputed_init,
            values=None,
            iteration=0)

        # Iterative functional updating
        for k in range(self._num_iters):
            updates = self.forward_local_updater(tr_reprs, data.tr_output, data.tr_input, iteration=k) #return negative u

            # Fourier features
            tr_input_ff = self.fourier_features(
                X=data.tr_input, 
                recompute=self._indp_iter, 
                precomputed=tr_input_ff,
                iteration=k)

            val_input_ff = self.fourier_features(
                X=data.val_input, 
                recompute=self._indp_iter, 
                precomputed=val_input_ff,
                iteration=k)

            precomputed = self.forward_kernel_or_attention_precompute(
                querys=tf.concat([tr_input_ff, val_input_ff],axis=-2), 
                keys=tr_input_ff,
                recompute=self._indp_iter,
                precomputed=precomputed,
                values=None,
                iteration=k)

            tr_updates, val_updates = tf.split(
                self.alpha * self.forward_kernel_or_attention(
                    querys=None, 
                    keys=None, 
                    precomputed=precomputed,
                    values=updates), 
                num_or_size_splits=[tf.shape(data.tr_input)[-2], tf.shape(data.val_input)[-2]], 
                axis=-2)
            tr_reprs += tr_updates
            val_reprs += val_updates

        # Decode functional representation and compute loss and metric
        classifier_weights, tr_orth = self.forward_decoder(tr_reprs)
        tr_loss, batch_tr_metric = self.predict_and_calculate_loss_and_acc(data.tr_input, data.tr_output, classifier_weights)
        classifier_weights, val_orth = self.forward_decoder(val_reprs)
        val_loss, batch_val_metric = self.predict_and_calculate_loss_and_acc(data.val_input, data.val_output, classifier_weights)


        # # Aggregate loss in a batch
        # batch_tr_loss = tf.math.reduce_mean(tr_loss)
        # batch_val_loss = tf.math.reduce_mean(val_loss)

        #Additional regularisation penalty
        #return batch_val_loss + self._decoder_orthogonality_reg, batch_tr_metric, batch_val_metric  #TODO:? need weights for l2

        return val_loss, tr_orth, batch_tr_metric, batch_val_metric
    

class MetaFunRegressorV2(MetaFunBaseV2, MetaFunRegressor):
    def __init__(self, config, data_source="regression", no_batch=False, name="MetaFunRegressorV2"):
        super(MetaFunRegressorV2, self).__init__(config=config, data_source=data_source, no_batch=no_batch, name=name)

    def __call__(self, data, is_training=tf.constant(True, dtype=tf.bool)):
        """
        data: dictionary-like form, with attributes "tr_input", "val_input", "tr_output", "val_output"
            Classification training/validation data of a task.
        is_training: bool
            If True, training mode
        """
        # Initialise Variables #TODO: is_training set as tf.constant in learner
        self.is_training.assign(is_training)
        assert self.has_initialised, "the learner is not initialised, please initialise via the method - .initialise"

        all_tr_reprs = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters+1)
        all_val_reprs = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters+1)

        # Initialise r
        tr_reprs = self.forward_initialiser(data.tr_input, is_training=self.is_training)
        val_reprs = self.forward_initialiser(data.val_input, is_training=self.is_training)
        
        all_tr_reprs = all_tr_reprs.write(0, tr_reprs)
        all_val_reprs = all_val_reprs.write(0, val_reprs)

        # Fourier features
        tr_input_ff = self.fourier_features(
            X=data.tr_input, 
            recompute=tf.math.logical_not(self._indp_iter), 
            precomputed=self.tr_data_ff_init,
            iteration=0)

        val_input_ff = self.fourier_features(
            X=data.val_input, 
            recompute=tf.math.logical_not(self._indp_iter), 
            precomputed=self.val_data_ff_init,
            iteration=0)

        # Precompute target context interaction for kernel/attention
        precomputed = self.forward_kernel_or_attention_precompute(
            querys=tf.concat([tr_input_ff, val_input_ff],axis=-2), 
            keys=tr_input_ff,
            recompute=tf.math.logical_not(self._indp_iter), # if not indepnedent iteration, precompute 
            precomputed=self.precomputed_init,
            values=None,
            iteration=0)

        # Iterative functional updating
        for k in range(self._num_iters):
            updates = self.forward_local_updater(r=tr_reprs, y=data.tr_output, x=data.tr_input, iteration=k)

            # Fourier features
            tr_input_ff = self.fourier_features(
                X=data.tr_input, 
                recompute=self._indp_iter, 
                precomputed=tr_input_ff,
                iteration=k)

            val_input_ff = self.fourier_features(
                X=data.val_input, 
                recompute=self._indp_iter, 
                precomputed=val_input_ff,
                iteration=k)

            precomputed = self.forward_kernel_or_attention_precompute(
                querys=tf.concat([tr_input_ff, val_input_ff],axis=-2), 
                keys=tr_input_ff,
                recompute=self._indp_iter,
                precomputed=precomputed,
                values=None,
                iteration=k)

            tr_updates, val_updates = tf.split(
                self.alpha * self.forward_kernel_or_attention(
                    querys=None,
                    keys=None,
                    precomputed=precomputed,
                    values=updates),
                num_or_size_splits=[tf.shape(data.tr_input)[-2], tf.shape(data.val_input)[-2]],
                axis=-2
                )
            tr_reprs += tr_updates
            val_reprs += val_updates
            all_tr_reprs = all_tr_reprs.write(1+k, tr_reprs)
            all_val_reprs = all_val_reprs.write(1+k, val_reprs)

        # Decode functional representation and compute loss and metric
        all_val_mu = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters+1)
        all_val_sigma = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters+1)
        all_tr_mu = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters+1)
        all_tr_sigma = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters+1)

        for k in range(self._num_iters+1): # store intermediate predictions
            weights = self.forward_decoder(all_tr_reprs.read(k))
            tr_mu, tr_sigma = self.predict(inputs=data.tr_input, weights=weights)
            weights = self.forward_decoder(all_val_reprs.read(k))
            val_mu, val_sigma = self.predict(inputs=data.val_input, weights=weights)
            all_val_mu = all_val_mu.write(k, val_mu)
            all_val_sigma = all_val_sigma.write(k, val_sigma)
            all_tr_mu = all_tr_mu.write(k, tr_mu)
            all_tr_sigma = all_tr_sigma.write(k, tr_sigma)

        tr_loss, tr_metric = self.calculate_loss_and_metrics(
            target_y=data.tr_output,
            mus=tr_mu,
            sigmas=tr_sigma)
        val_loss, val_metric = self.calculate_loss_and_metrics(
            target_y=data.val_output,
            mus=val_mu,
            sigmas=val_sigma
        )

        all_val_mu = all_val_mu.stack()
        all_val_sigma = all_val_sigma.stack()
        all_tr_mu = all_tr_mu.stack()
        all_tr_sigma = all_tr_sigma.stack()

        additional_loss = tf.constant(0., dtype=self._float_dtype)

        return val_loss, additional_loss, tr_metric, val_metric, all_val_mu, all_val_sigma, all_tr_mu, all_tr_sigma




if __name__ == "__main__":
    from utils import parse_config
    import os
    import numpy as np
    import collections
    config = parse_config(os.path.join(os.path.dirname(__file__),"config/debug.yaml"))
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
    module = MetaFunClassifier(config=config)
    dataloader = DataProvider("trial", config)
    dat = dataloader.generate()
    for i in dat.take(1):
        module.initialise(i)

    @tf.function
    def trial(x, is_training=tf.constant(True,tf.bool)):
        l,_,_,_ = module(x, is_training=is_training)
        return l

    print("DEBUGGGGGGGGGGGGGGG")
    for i in dat.take(1):
        print(trial(i, is_training=False))
        print(module(i, is_training=False)[0])

    trial(i, is_training=True)
    module(i, is_training=True)[0]


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
    # data_reg = ClassificationDescription(
    # tf.constant(np.random.random([2, 10,10]),dtype=tf.float32),
    # tf.constant(np.random.random([2,10,1]),dtype=tf.float32),
    # tf.constant(np.random.random([2, 10,10]),dtype=tf.float32),
    # tf.constant(np.random.random([2,10,1]),dtype=tf.float32))
    # @tf.function
    # def trial(x, is_training=True):
    #     l,*_ = module2(x, is_training=is_training)
    #     return l

    # print("DEBUGGGGGGGGGGGGGGG")
 
    # print(trial(data_reg, is_training=False))
    # print(module2(data_reg, is_training=False)[0])

    # print("Classification")
    # from data.leo_imagenet import DataProvider
    # module = MetaFunClassifierV2(config=config)
    # dataloader = DataProvider("trial", config)
    # dat = dataloader.generate()
    # for i in dat.take(1):
    #     module.initialise(i)

    # @tf.function
    # def trial(x, is_training=tf.constant(True,tf.bool)):
    #     l,_,_,_ = module(x, is_training=is_training)
    #     return l

    # print("DEBUGGGGGGGGGGGGGGG")
    # for i in dat.take(1):
    #     print(trial(i, is_training=False))
    #     print(module(i, is_training=False)[0])

    # trial(i, is_training=True)
    # module(i, is_training=True)[0]


    # print("Regression")
    # module2 = MetaFunRegressorV2(config=config)
    # ClassificationDescription = collections.namedtuple(
    # "ClassificationDescription",
    # ["tr_input", "tr_output", "val_input", "val_output"])
    
    # data_reg = ClassificationDescription(
    # tf.constant(np.random.random([2, 10,10]),dtype=tf.float32),
    # tf.constant(np.random.random([2,10,1]),dtype=tf.float32),
    # tf.constant(np.random.random([2, 10,10]),dtype=tf.float32),
    # tf.constant(np.random.random([2,10,1]),dtype=tf.float32))

    # module2.initialise(data_reg)
    # data_reg = ClassificationDescription(
    # tf.constant(np.random.random([2, 10,10]),dtype=tf.float32),
    # tf.constant(np.random.random([2,10,1]),dtype=tf.float32),
    # tf.constant(np.random.random([2, 10,10]),dtype=tf.float32),
    # tf.constant(np.random.random([2,10,1]),dtype=tf.float32))
    # @tf.function
    # def trial(x, is_training=True):
    #     l,*_ = module2(x, is_training=is_training)
    #     return l

    # print("DEBUGGGGGGGGGGGGGGG")
 
    # print(trial(data_reg, is_training=False))
    # print(module2(data_reg, is_training=False)[0])


