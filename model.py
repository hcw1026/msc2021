from functools import partial
import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt
import submodules as submodules
import utils
import copy


class MetaFunBase(snt.Module):
    def __init__(self, config, no_batch, num_classes=1, disable_decoder=False, name="MetaFun"):
        super(MetaFunBase, self).__init__(name=name)
        self._float_dtype = tf.float32
        self._int_dtype = tf.int32
        self.config = copy.deepcopy(config)
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
        self._kernel_lengthscale_trainable = _config["kernel_lengthscale_trainable"]
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
        self._nonlinearity = tf.nn.relu if config["Model"]["other"]["nonlinearity"] == "relu" else partial(tf.nn.leaky_relu(), alpha=0.1)
        self.initialiser = tf.keras.initializers.GlorotUniform()
        self.dropout = tf.keras.layers.Dropout(self._dropout_rate)
        self._no_batch = no_batch

        # Constant Initialization
        self._orthogonality_reg = tf.constant(0., dtype=self._float_dtype)
        self.is_training = tf.Variable(True, dtype=tf.bool, trainable=False)
        self.epoch = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.embedding_dim = tf.constant([1])
        self.has_initialised = False

        # Classification or Regression
        self._num_classes = num_classes
        self._repr_as_inputs = config["Model"]["comp"]["repr_as_inputs"]
        self._neural_updater_concat_x = config["Model"]["comp"]["neural_updater_concat_x"]
        self._stddev_const_scale = tf.constant(config["Model"]["comp"]["stddev_const_scale"], dtype=tf.float32)
        self._stddev_offset = tf.constant(config["Model"]["comp"]["stddev_const_scale"], dtype=tf.float32)
        self._fixed_sigma_value = tf.constant(config["Model"]["comp"]["fixed_sigma_value"], dtype=tf.float32)
        self._fixed_sigma_epoch = tf.constant(config["Model"]["comp"]["fixed_sigma_epoch"], dtype=tf.int32)

        # Define metric
        self._metric_names = []

        if self._no_decoder:
            self._dim_reprs = 1

    @snt.once
    def additional_initialise_after(self, data_instance=None): # for extra initialisation steps
        pass

    @snt.once
    def additional_initialise_pre(self, data_instance=None): # for extra initialisation steps
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
        self.sample_tr_data = data_instance.tr_input
        self.sample_val_data = data_instance.val_input
        self.additional_initialise_pre(data_instance=data_instance)

        self.forward_initialiser = self.forward_initialiser_base()
        self.forward_local_updater = self.forward_local_updater_base()
        self.forward_decoder = self.forward_decoder_base()
        self.forward_kernel_or_attention_precompute, self.forward_kernel_or_attention = self.forward_kernel_or_attention_base()

        # Extra internal functions
        self.predict_init()
        self.sample_init()

        # Change initialisation state
        self.additional_initialise_after(data_instance=data_instance)
        self.has_initialised = True
        #self.__call__(data_instance)

        # Regularisation variables
        self.regularise_variables = utils.get_linear_layer_variables(self)

    @snt.once
    def additional_initialise_after(self, data_instance=None):
        """for data-specific initialisation and any extra initialisation"""

        # Deduce precompute shape of forward_kernel_or_attention_precompute
        self.precomputed_init = tf.zeros_like(self.forward_kernel_or_attention_precompute(
            querys=tf.concat([self.sample_tr_data, self.sample_val_data], axis=-2),
            keys=self.sample_tr_data,
            values=None,
            recompute=True,
            precomputed=None,
            iteration=0,
            is_training=self.is_training
        ))

    def __call__(self, data, is_training=tf.constant(True, dtype=tf.bool), epoch=tf.constant(0, dtype=tf.int32)):
        """
        data: dictionary-like form, with attributes "tr_input", "val_input", "tr_output", "val_output"
            Classification training/validation data of a task.
        is_training: bool
            If True, training mode
        """
        # Initialise Variables #TODO: is_training set as tf.constant in learner
        self.is_training.assign(is_training)
        self.epoch.assign(epoch)
        assert self.has_initialised, "the learner is not initialised, please initialise via the method - .initialise"

        tr_input, val_input, tr_output, val_output = data.tr_input, data.val_input, data.tr_output, data.val_output
        tr_reprs, val_reprs = self.Encoder(tr_input=tr_input, val_input=val_input, tr_output=tr_output)
        return self.Decoder(data=data, tr_reprs=tr_reprs, val_reprs=val_reprs, tr_input=data.tr_input, val_input=data.val_input)

    def Encoder(self, tr_input, val_input, tr_output):
        # Initialise r
        tr_reprs = self.forward_initialiser(tr_input, is_training=self.is_training)
        val_reprs = self.forward_initialiser(val_input, is_training=self.is_training)

        # Precompute target context interaction for kernel/attention
        precomputed = self.forward_kernel_or_attention_precompute(
            querys=tf.concat([tr_input, val_input],axis=-2), 
            keys=tr_input,
            recompute=tf.math.logical_not(self._indp_iter), # if not indepnedent iteration, precompute 
            precomputed=self.precomputed_init,
            values=None,
            iteration=0,
            is_training=self.is_training)

        # Iterative functional updating
        for k in range(self._num_iters):
            updates = self.forward_local_updater(tr_reprs, tr_output, tr_input, iteration=k) #return negative u

            precomputed = self.forward_kernel_or_attention_precompute(
                querys=tf.concat([tr_input, val_input],axis=-2), 
                keys=tr_input,
                recompute=self._indp_iter,
                precomputed=precomputed,
                values=None,
                iteration=k,
                is_training=self.is_training)

            tr_updates, val_updates = tf.split(
                self.alpha * self.forward_kernel_or_attention(
                    querys=None, 
                    keys=None, 
                    precomputed=precomputed,
                    values=updates), 
                num_or_size_splits=[tf.shape(tr_input)[-2], tf.shape(val_input)[-2]], 
                axis=-2)
            tr_reprs += tr_updates
            val_reprs += val_updates

        return tr_reprs, val_reprs

    def Decoder(self, data, tr_reprs, val_reprs, tr_input, val_input):
        # Decode functional representation and compute loss and metric
        classifier_weights, tr_orth = self.forward_decoder(tr_reprs)
        tr_loss, batch_tr_metric = self.predict_and_calculate_loss_and_acc(tr_input, data.tr_output, classifier_weights)
        classifier_weights, val_orth = self.forward_decoder(val_reprs)
        val_loss, batch_val_metric = self.predict_and_calculate_loss_and_acc(val_input, data.val_output, classifier_weights)


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


    def constant_initialiser(self, x, is_training=False):
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
            trainable=self._kernel_lengthscale_trainable,
            name="kernel_lengthscale"
        )

    @snt.once
    def se_kernel_init(self):
        self.se_kernel_all_init()
        self._se_kernel = submodules.squared_exponential_kernel(complete_return=False)

    def se_kernel_precompute(self, querys, keys, recompute, precomputed, values=None, iteration=0, is_training=False):
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

    def deep_se_kernel_precompute(self, querys, keys, recompute, precomputed, values=None, iteration=0, is_training=False):
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

    def attention_block_precompute(self, querys, keys, recompute, precomputed, values=None, iteration=0, is_training=False):
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
        return tf.keras.losses.CategoricalCrossentropy(
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
        self._loss_type = config["Train"]["loss_type"]
        self._decoder_output_sizes_f = config["Model"]["arch"]["decoder_output_sizes"].copy() if config["Model"]["arch"]["decoder_output_sizes"] is not None else config["Model"]["arch"]["decoder_output_sizes"]

        self._metric_names = ["mse", "logprob"] # must be in the order of metric output

    @snt.once
    def initialise(self, data_instance):
        """initialiser variables and functions"""

        self.embedding_dim = data_instance.tr_input.get_shape()[-1]
        self.sample_tr_data = data_instance.tr_input
        self.sample_val_data = data_instance.val_input
        self.output_dim = data_instance.tr_output.get_shape()[-1]
        self._decoder_output_sizes = [self._nn_size] * (self._nn_layers-1) if self._decoder_output_sizes_f is None else self._decoder_output_sizes_f
        self._decoder_output_sizes += [self.output_dim*2]
        self.additional_initialise_pre(data_instance=data_instance)

        self.forward_initialiser = self.forward_initialiser_base()
        self.forward_local_updater = self.forward_local_updater_base()
        self.forward_decoder = self.forward_decoder_base()
        self.forward_kernel_or_attention_precompute, self.forward_kernel_or_attention = self.forward_kernel_or_attention_base()

        # Extra internal functions
        self.predict_init()
        self.calculate_loss_and_metrics_init()
        self.sample_init()

        # Change initialisation state
        self.additional_initialise_after(data_instance=data_instance)
        self.has_initialised = True
        #self.__call__(data_instance)

        # Regularisation variables
        self.regularise_variables = utils.get_linear_layer_variables(self)

    @snt.once
    def additional_initialise_after(self):
        """for data-specific initialisation and any extra initialisation"""

        # Deduce precompute shape of forward_kernel_or_attention_precompute
        self.precomputed_init = tf.zeros_like(self.forward_kernel_or_attention_precompute(
            querys=tf.concat([self.sample_tr_data, self.sample_val_data], axis=-2),
            keys=self.sample_tr_data,
            values=None,
            recompute=True,
            precomputed=None,
            iteration=0,
            is_training=self.is_training
        ))

    def __call__(self, data, is_training=tf.constant(True, dtype=tf.bool), epoch=tf.constant(1, dtype=tf.int32)):
        """
        data: dictionary-like form, with attributes "tr_input", "val_input", "tr_output", "val_output"
            Classification training/validation data of a task.
        is_training: bool
            If True, training mode
        """
        # Initialise Variables #TODO: is_training set as tf.constant in learner
        self.is_training.assign(is_training)
        self.epoch.assign(epoch)
        assert self.has_initialised, "the learner is not initialised, please initialise via the method - .initialise"

        tr_input, val_input, tr_output, val_output = data.tr_input, data.val_input, data.tr_output, data.val_output
        all_tr_reprs, all_val_reprs = self.Encoder(tr_input=tr_input, val_input=val_input, tr_output=tr_output)
        return self.Decoder(data=data, all_tr_reprs=all_tr_reprs, all_val_reprs=all_val_reprs, epoch=epoch)

    def Encoder(self, tr_input, val_input, tr_output):
        all_tr_reprs = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters+1)
        all_val_reprs = tf.TensorArray(dtype=self._float_dtype, size=self._num_iters+1)

        # Initialise r
        tr_reprs = self.forward_initialiser(tr_input, is_training=self.is_training)
        val_reprs = self.forward_initialiser(val_input, is_training=self.is_training)
        
        all_tr_reprs = all_tr_reprs.write(0, tr_reprs)
        all_val_reprs = all_val_reprs.write(0, val_reprs)

        # Precompute target context interaction for kernel/attention
        precomputed = self.forward_kernel_or_attention_precompute(
            querys=tf.concat([tr_input, val_input],axis=-2), 
            keys=tr_input,
            recompute=tf.math.logical_not(self._indp_iter),
            precomputed=self.precomputed_init,
            values=None,
            iteration=0,
            is_training=self.is_training)

        # Iterative functional updating
        for k in range(self._num_iters):
            updates = self.forward_local_updater(r=tr_reprs, y=tr_output, x=tr_input, iteration=k)

            precomputed = self.forward_kernel_or_attention_precompute(
                querys=tf.concat([tr_input, val_input],axis=-2), 
                keys=tr_input,
                recompute=self._indp_iter,
                precomputed=precomputed,
                values=None,
                iteration=k,
                is_training=self.is_training)

            tr_updates, val_updates = tf.split(
                self.alpha * self.forward_kernel_or_attention(
                    querys=None,
                    keys=None,
                    precomputed=precomputed,
                    values=updates),
                num_or_size_splits=[tf.shape(tr_input)[-2], tf.shape(val_input)[-2]],
                axis=-2
                )
            tr_reprs += tr_updates
            val_reprs += val_updates
            all_tr_reprs = all_tr_reprs.write(1+k, tr_reprs)
            all_val_reprs = all_val_reprs.write(1+k, val_reprs)
            return all_tr_reprs, all_val_reprs

    def Decoder(self, data, all_tr_reprs, all_val_reprs, epoch=tf.constant(1, dtype=tf.int32)):
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

            if epoch <= self._fixed_sigma_epoch:
                tr_sigma = self._fixed_sigma_value
                val_sigma = self._fixed_sigma_value
            else:
                tr_sigma = tr_sigma
                val_sigma = val_sigma

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
            xNone=not self._neural_updater_concat_x,
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
            trainable=self._kernel_lengthscale_trainable,
            name="kernel_lengthscale"
        )

    @snt.once
    def se_kernel_init(self):
        self.se_kernel_all_init()
        self._se_kernel = submodules.squared_exponential_kernel(complete_return=False)
    
    def se_kernel_precompute(self, querys, keys, recompute, precomputed, values=None, iteration=0, is_training=False):
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

    def deep_se_kernel_precompute(self, querys, keys, recompute, precomputed, values=None, iteration=0, is_training=False):
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

    def attention_block_precompute(self, querys, keys, recompute, precomputed, values=None, iteration=0, is_training=False):
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

        def predict_repr_as_inputs_simple(inputs, weights):
            preds = self._predict_repr_as_inputs_simple(inputs=inputs, weights=weights)
            return _split(preds)

        def predict_not_repr_as_inputs(inputs, weights):
            preds = self.custom_MLP(inputs=inputs, weights=weights)
            return _split(preds)
        
        def _split(preds):
            mu, log_sigma = tf.split(preds, 2, axis=-1)
            sigma = self._stddev_const_scale + (1-self._stddev_const_scale) * tf.nn.softplus(log_sigma-self._stddev_offset)
            return mu, sigma

        if self._no_decoder:
            if self._dim_reprs == 1:
                self._predict = predict_no_decoder1
            elif self._dim_reprs == 2:
                self._predict = predict_no_decoder2
            else:
                raise Exception("num_reprs must <=2 if no_decoder")
        elif self._repr_as_inputs is True:
            self._predict_repr_as_inputs = submodules.predict_repr_as_inputs(
                output_sizes=self._decoder_output_sizes,
                initialiser=self.initialiser,
                nonlinearity=self._nonlinearity
            )
            self._predict =  predict_repr_as_inputs
        elif self._repr_as_inputs == "simple":
            self._predict_repr_as_inputs_simple = submodules.predict_repr_as_inputs_simple(
                nn_size=self._nn_size,
                nn_layers=self._nn_layers, 
                output_dim=self.output_dim,
                initialiser=self.initialiser,
                nonlinearity=self._nonlinearity
            )
            self._predict =  predict_repr_as_inputs_simple
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
    def additional_initialise_pre(self, data_instance=None):
        if self._use_ff:
            self.fourier_features_init()
            self.fourier_features = self.fourier_features_compute
        else:
            self.fourier_features = lambda X, recompute, precomputed, iteration: X

        self.sample_tr_data = self.fourier_features(X=self.sample_tr_data, recompute=True, precomputed=None, iteration=0)
        self.sample_val_data = self.fourier_features(X=self.sample_val_data, recompute=True, precomputed=None, iteration=0)

    @snt.once
    def additional_initialise_after(self, data_instance=None):
        # Deduce precompute shape of forward_kernel_or_attention_precompute
        self.precomputed_init = tf.zeros_like(self.forward_kernel_or_attention_precompute(
            querys=tf.concat([self.sample_tr_data, self.sample_val_data], axis=-2),
            keys=self.sample_tr_data,
            values=None,
            recompute=True,
            precomputed=None,
            iteration=0,
            is_training=self.is_training
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

    def forward_kernel_or_attention_base(self):
        """functional pooling"""
        if self._use_kernel:
            if self._kernel_type == tf.constant("se", dtype=tf.string):
                self.se_kernel_init()
                return (self.se_kernel_precompute, self.se_kernel_backend)
            elif self._kernel_type == tf.constant("deep_se", dtype=tf.string):
                self.deep_se_kernel_init()
                return (self.deep_se_kernel_precompute, self.deep_se_kernel_backend)
            elif self._kernel_type == tf.constant("rff", dtype=tf.string):
                self.rff_kernel_init()
                return (self.rff_kernel_precompute, self.rff_kernel_backend)
            else:
                raise NameError("Unknown kernel type")
        else:
            self.attention_block_init()
            return (self.attention_block_precompute, self.attention_block_backend)


class MetaFunClassifierV2(MetaFunBaseV2, MetaFunClassifier):
    def __init__(self, config, data_source="leo_imagenet", no_batch=False, name="MetaFunClassifierV2", **kwargs):
        self._num_classes = config["Data"][data_source]["num_classes"]
        super(MetaFunClassifierV2, self).__init__(config=config, data_source=data_source, num_classes=self._num_classes, no_batch=no_batch, name=name)

    def __call__(self, data, is_training=tf.constant(True, dtype=tf.bool), epoch=tf.constant(0, dtype=tf.int32)):
        """
        data: dictionary-like form, with attributes "tr_input", "val_input", "tr_output", "val_output"
            Classification training/validation data of a task.
        is_training: bool
            If True, training mode
        """
        # Initialise Variables #TODO: is_training set as tf.constant in learner
        self.is_training.assign(is_training)
        self.epoch.assign(epoch)
        assert self.has_initialised, "the learner is not initialised, please initialise via the method - .initialise"

        tr_input, val_input, tr_output, val_output = data.tr_input, data.val_input, data.tr_output, data.val_output
        tr_reprs, val_reprs, tr_input_ff, val_input_ff = self.Encoder(tr_input=tr_input, val_input=val_input, tr_output=tr_output)
        return self.Decoder(data=data, tr_reprs=tr_reprs, val_reprs=val_reprs, tr_input=tr_input_ff, val_input=val_input_ff, epoch=epoch)

    def Encoder(self, tr_input, val_input, tr_output):
        # Initialise r
        tr_reprs = self.forward_initialiser(tr_input, is_training=self.is_training)
        val_reprs = self.forward_initialiser(val_input, is_training=self.is_training)

        # Fourier features
        tr_input_ff = self.fourier_features(
            X=tr_input, 
            recompute=tf.math.logical_not(self._indp_iter), 
            precomputed=self.sample_tr_data,
            iteration=0)

        val_input_ff = self.fourier_features(
            X=val_input, 
            recompute=tf.math.logical_not(self._indp_iter), 
            precomputed=self.sample_val_data,
            iteration=0)

        # Precompute target context interaction for kernel/attention
        precomputed = self.forward_kernel_or_attention_precompute(
            querys=tf.concat([tr_input_ff, val_input_ff],axis=-2), 
            keys=tr_input_ff,
            recompute=tf.math.logical_not(self._indp_iter), # if not indepnedent iteration, precompute 
            precomputed=self.precomputed_init,
            values=None,
            iteration=0,
            is_training=self.is_training)

        # Iterative functional updating
        for k in range(self._num_iters):
            updates = self.forward_local_updater(tr_reprs, tr_output, tr_input, iteration=k) #return negative u

            # Fourier features
            tr_input_ff = self.fourier_features(
                X=tr_input, 
                recompute=self._indp_iter, 
                precomputed=tr_input_ff,
                iteration=k)

            val_input_ff = self.fourier_features(
                X=val_input, 
                recompute=self._indp_iter, 
                precomputed=val_input_ff,
                iteration=k)

            precomputed = self.forward_kernel_or_attention_precompute(
                querys=tf.concat([tr_input_ff, val_input_ff],axis=-2), 
                keys=tr_input_ff,
                recompute=self._indp_iter,
                precomputed=precomputed,
                values=None,
                iteration=k,
                is_training=self.is_training)

            tr_updates, val_updates = tf.split(
                self.alpha * self.forward_kernel_or_attention(
                    querys=None, 
                    keys=None, 
                    precomputed=precomputed,
                    values=updates), 
                num_or_size_splits=[tf.shape(tr_input)[-2], tf.shape(val_input)[-2]], 
                axis=-2)
            tr_reprs += tr_updates
            val_reprs += val_updates
        
        return tr_reprs, val_reprs, tr_input_ff, val_input_ff
    

class MetaFunRegressorV2(MetaFunBaseV2, MetaFunRegressor):
    def __init__(self, config, data_source="regression", no_batch=False, name="MetaFunRegressorV2", **kwargs):
        super(MetaFunRegressorV2, self).__init__(config=config, data_source=data_source, no_batch=no_batch, name=name)

    def __call__(self, data, is_training=tf.constant(True, dtype=tf.bool), epoch=tf.constant(0, dtype=tf.int32)):
        """
        data: dictionary-like form, with attributes "tr_input", "val_input", "tr_output", "val_output"
            Classification training/validation data of a task.
        is_training: bool
            If True, training mode
        """
        # Initialise Variables #TODO: is_training set as tf.constant in learner
        self.is_training.assign(is_training)
        self.epoch.assign(epoch)
        assert self.has_initialised, "the learner is not initialised, please initialise via the method - .initialise"
    
        tr_input, val_input, tr_output, val_output = data.tr_input, data.val_input, data.tr_output, data.val_output
        tr_reprs, val_reprs, tr_input_ff, val_input_ff = self.Encoder(tr_input=tr_input, val_input=val_input, tr_output=tr_output)
        return self.Decoder(data=data, tr_reprs=tr_reprs, val_reprs=val_reprs, tr_input=tr_input_ff, val_input=val_input_ff, epoch=epoch)

    def Encoder(self, tr_input, val_input, tr_output):

        # Initialise r
        tr_reprs = self.forward_initialiser(tr_input, is_training=self.is_training)
        val_reprs = self.forward_initialiser(val_input, is_training=self.is_training)
        
        # Fourier features
        tr_input_ff = self.fourier_features(
            X=tr_input, 
            recompute=tf.math.logical_not(self._indp_iter), 
            precomputed=self.sample_tr_data,
            iteration=0)

        val_input_ff = self.fourier_features(
            X=val_input, 
            recompute=tf.math.logical_not(self._indp_iter), 
            precomputed=self.sample_val_data,
            iteration=0)

        # Precompute target context interaction for kernel/attention
        precomputed = self.forward_kernel_or_attention_precompute(
            querys=tf.concat([tr_input_ff, val_input_ff],axis=-2), 
            keys=tr_input_ff,
            recompute=tf.math.logical_not(self._indp_iter), # if not indepnedent iteration, precompute 
            precomputed=self.precomputed_init,
            values=None,
            iteration=0,
            is_training=self.is_training)


        # Iterative functional updating
        for k in range(self._num_iters):
            updates = self.forward_local_updater(r=tr_reprs, y=tr_output, x=tr_input, iteration=k)

            # Fourier features
            tr_input_ff = self.fourier_features(
                X=tr_input, 
                recompute=self._indp_iter, 
                precomputed=tr_input_ff,
                iteration=k)

            val_input_ff = self.fourier_features(
                X=val_input, 
                recompute=self._indp_iter, 
                precomputed=val_input_ff,
                iteration=k)

            precomputed = self.forward_kernel_or_attention_precompute(
                querys=tf.concat([tr_input_ff, val_input_ff],axis=-2), 
                keys=tr_input_ff,
                recompute=self._indp_iter,
                precomputed=precomputed,
                values=None,
                iteration=k,
                is_training=self.is_training)

            tr_updates, val_updates = tf.split(
                self.alpha * self.forward_kernel_or_attention(
                    querys=None,
                    keys=None,
                    precomputed=precomputed,
                    values=updates),
                num_or_size_splits=[tf.shape(tr_input)[-2], tf.shape(val_input)[-2]],
                axis=-2
                )

            tr_reprs += tr_updates
            val_reprs += val_updates

        return tr_reprs, val_reprs, tr_input_ff, val_input_ff

    def Decoder(self, data, tr_reprs, val_reprs, tr_input, val_input, epoch=tf.constant(1, dtype=tf.int32)):
        weights = self.forward_decoder(tr_reprs)
        tr_mu, tr_sigma = self.predict(inputs=tr_input, weights=weights)
        weights = self.forward_decoder(val_reprs)
        val_mu, val_sigma = self.predict(inputs=val_input, weights=weights)

        if epoch <= self._fixed_sigma_epoch:
            tr_sigma = self._fixed_sigma_value
            val_sigma = self._fixed_sigma_value
        else:
            tr_sigma = tr_sigma
            val_sigma = val_sigma


        tr_loss, tr_metric = self.calculate_loss_and_metrics(
            target_y=data.tr_output,
            mus=tr_mu,
            sigmas=tr_sigma)
        val_loss, val_metric = self.calculate_loss_and_metrics(
            target_y=data.val_output,
            mus=val_mu,
            sigmas=val_sigma
        )

        additional_loss = tf.constant(0., dtype=self._float_dtype)
        return val_loss, additional_loss, tr_metric, val_metric, val_mu, val_sigma, tr_mu, tr_sigma


class MetaFunRegressorV3(MetaFunRegressorV2):
    def __init__(self, config, data_source="regression", no_batch=False, name="MetaFunRegressorV3", **kwargs):
        super(MetaFunRegressorV3, self).__init__(config=config, data_source=data_source, no_batch=no_batch, name=name)
        _config = config["Model"]["rff"]
        self._rff_dim_init = _config["dim_init"]
        self._rff_n_layers = _config["n_layers"]
        self._rff_perm_equi_pool_fn = _config["perm_equi_pool_fn"]
        self._rff_mapping = _config["mapping"]
        self._rff_dim_pre_tr = _config["dim_pre_tr"]
        self._rff_sab_nn_layers = _config["sab_nn_layers"]
        self._rff_sab_num_heads = _config["sab_num_heads"]
        self._rff_isab_n_induce_points = _config["isab_n_induce_points"]
        self._rff_init_trainable = _config["init_trainable"]
        self._rff_init_distr = _config["init_distr"]
        self._rff_init_distr_param = _config["init_distr_param"]
        self._rff_weight_trainable = _config["weight_trainable"]
        self._rff_transform_dim = _config["transform_dim"]
        self._rff_dropout_rate = _config["dropout_rate"]

    @snt.once
    def predict_init(self):
        """ backend of decoder to produce mean and variance of predictions"""
        def predict_no_decoder1(inputs, weights):
            return weights, tf.ones_like(weights) * 0.5

        def predict_no_decoder2(inputs, weights):
            return tf.split(weights, 2, axis=-1)

        def predict_repr_as_inputs(inputs, weights):
            preds = self._predict_repr_as_inputs(inputs=tf.zeros_like(inputs), weights=weights)
            return _split(preds)

        def predict_repr_as_inputs_simple(inputs, weights):
            preds = self._predict_repr_as_inputs_simple(inputs=tf.zeros_like(inputs), weights=weights)
            return _split(preds)

        def predict_not_repr_as_inputs(inputs, weights):
            preds = self.custom_MLP(inputs=tf.zeros_like(inputs), weights=weights)
            return _split(preds)
        
        def _split(preds):
            mu, log_sigma = tf.split(preds, 2, axis=-1)
            sigma = self._stddev_const_scale + (1-self._stddev_const_scale) * tf.nn.softplus(log_sigma-self._stddev_offset)
            return mu, sigma

        if self._no_decoder:
            if self._dim_reprs == 1:
                self._predict = predict_no_decoder1
            elif self._dim_reprs == 2:
                self._predict = predict_no_decoder2
            else:
                raise Exception("num_reprs must <=2 if no_decoder")
        elif self._repr_as_inputs is True:
            self._predict_repr_as_inputs = submodules.predict_repr_as_inputs(
                output_sizes=self._decoder_output_sizes,
                initialiser=self.initialiser,
                nonlinearity=self._nonlinearity
            )
            self._predict =  predict_repr_as_inputs
        elif self._repr_as_inputs == "simple":
            self._predict_repr_as_inputs_simple = submodules.predict_repr_as_inputs_simple(
                nn_size=self._nn_size,
                nn_layers=self._nn_layers, 
                output_dim=self.output_dim,
                initialiser=self.initialiser,
                nonlinearity=self._nonlinearity
            )
            self._predict =  predict_repr_as_inputs_simple
        else:
            self.custom_MLP = submodules.custom_MLP(
                output_sizes=self._decoder_output_sizes,
                embedding_dim=self.embedding_dim,
                nonlinearity=self._nonlinearity)
            self._predict =  predict_not_repr_as_inputs

    @snt.once
    def rff_kernel_init(self):

        if self._rff_mapping == "deepset1":
            mapping = partial(submodules.DeepSet,
                dim_out=self.embedding_dim if self._rff_transform_dim is None else self._rff_transform_dim, #dimension of each random feature w
                n_layers=self._rff_n_layers,
                dim_pre_tr=self._rff_dim_pre_tr,
                pool_fn=self._rff_perm_equi_pool_fn,
                initialiser=self.initialiser)
        elif self._rff_mapping == None:
            mapping = None
        elif self._rff_mapping == "SAB":
            mapping = partial(submodules.SAB,
                dim_out=self.embedding_dim if self._rff_transform_dim is None else self._rff_transform_dim,
                n_layers=self._rff_n_layers,
                dim_pre_tr=self._rff_dim_pre_tr,
                nn_layers=self._rff_sab_nn_layers,
                num_heads=self._rff_sab_num_heads,
                initialiser=self.initialiser,
                nonlinearity=self._nonlinearity,
                float_dtype=self._float_dtype
                )
        elif self._rff_mapping == "ISAB":
            mapping = partial(submodules.ISAB,
                dim_out=self.embedding_dim if self._rff_transform_dim is None else self._rff_transform_dim,
                n_layers=self._rff_n_layers,
                dim_pre_tr=self._rff_dim_pre_tr,
                n_induce_points=self._rff_isab_n_induce_points,
                nn_layers=self._rff_sab_nn_layers,
                num_heads=self._rff_sab_num_heads,
                initialiser=self.initialiser,
                nonlinearity=self._nonlinearity,
                float_dtype=self._float_dtype
                )
        else:
            raise NameError("unknown rff mapping")
        self._rff_kernel = submodules.rff_kernel(
            dim_init=self._rff_dim_init,
            mapping=mapping,
            embedding_dim=self.embedding_dim,
            transform_dim=self._rff_transform_dim,
            initialiser=self.initialiser,
            init_distr=self._rff_init_distr, 
            init_distr_param=self._rff_init_distr_param,
            float_dtype=self._float_dtype,
            rff_init_trainable=self._rff_init_trainable,
            rff_weight_trainable=self._rff_weight_trainable,
            num_iters=self._num_iters,
            indp_iter=self._indp_iter,
            complete_return=False,
            dropout_rate=self._rff_dropout_rate
            )

    def rff_kernel_precompute(self, querys, keys, recompute, precomputed, values=None, iteration=0, is_training=False):
        return self._rff_kernel(querys=querys, keys=keys, recompute=recompute, precomputed=precomputed, values=values, iteration=iteration, is_training=is_training)

    def rff_kernel_backend(self, querys, keys, precomputed, values):
        return self._rff_kernel.backend(query_key=precomputed, values=values)


class MetaFunRegressorV3b(MetaFunRegressorV3):
    def __init__(self, config, data_source="regression", no_batch=False, name="MetaFunRegressorV3b", **kwargs):
        super(MetaFunRegressorV3b, self).__init__(config=config, data_source=data_source, no_batch=no_batch, name=name)

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

        def predict_repr_as_inputs_simple(inputs, weights):
            tf.print(inputs.shape, weights.shape)
            weights_concat = tf.concat([inputs, weights], axis=-1)
            preds = self._predict_repr_as_inputs_simple(inputs=inputs, weights=weights_concat) #inputs are disabled 
            return _split(preds)

        def predict_not_repr_as_inputs(inputs, weights):
            preds = self.custom_MLP(inputs=inputs, weights=weights)
            return _split(preds)
        
        def _split(preds):
            mu, log_sigma = tf.split(preds, 2, axis=-1)
            sigma = self._stddev_const_scale + (1-self._stddev_const_scale) * tf.nn.softplus(log_sigma-self._stddev_offset)
            return mu, sigma

        if self._no_decoder:
            if self._dim_reprs == 1:
                self._predict = predict_no_decoder1
            elif self._dim_reprs == 2:
                self._predict = predict_no_decoder2
            else:
                raise Exception("num_reprs must <=2 if no_decoder")
        elif self._repr_as_inputs is True:
            self._predict_repr_as_inputs = submodules.predict_repr_as_inputs(
                output_sizes=self._decoder_output_sizes,
                initialiser=self.initialiser,
                nonlinearity=self._nonlinearity
            )
            self._predict =  predict_repr_as_inputs
        elif self._repr_as_inputs == "simple":
            self._predict_repr_as_inputs_simple = submodules.predict_repr_as_inputs_simple(
                nn_size=self._nn_size,
                nn_layers=self._nn_layers, 
                output_dim=self.output_dim,
                initialiser=self.initialiser,
                nonlinearity=self._nonlinearity
            )
            self._predict =  predict_repr_as_inputs_simple
        else:
            self.custom_MLP = submodules.custom_MLP(
                output_sizes=self._decoder_output_sizes,
                embedding_dim=self.embedding_dim,
                nonlinearity=self._nonlinearity)
            self._predict =  predict_not_repr_as_inputs


class MetaFunBaseGLV2(MetaFunBaseV2):
    def __init__(self, config, no_batch, num_classes=1, name="MetaFunGLV2", **kwargs):
        super(MetaFunBaseGLV2, self).__init__(config=config, no_batch=no_batch, num_classes=num_classes, name=name)

    @snt.once
    def additional_initialise_after(self, data_instance=None):
        # Deduce precompute shape of forward_kernel_or_attention_precompute
        self.precomputed_init = tf.zeros_like(self.forward_kernel_or_attention_precompute(
            querys=tf.concat([self.sample_tr_data, self.sample_val_data], axis=-2),
            keys=self.sample_tr_data,
            values=None,
            recompute=True,
            precomputed=None,
            iteration=0,
            is_training=self.is_training
        ))

        self.sample_latent_init()
        self.latent_decoder_init()


class MetaFunRegressorGLV3(MetaFunBaseGLV2, MetaFunRegressorV3):
    def __init__(self, config, data_source="regression", no_batch=False, name="MetaFunRegressorGLV3", **kwargs):
        super(MetaFunRegressorGLV3, self).__init__(config=config, data_source=data_source, no_batch=no_batch, name=name)
        _config = config["Model"]["latent"]
        self._num_z_samples = _config["num_z_samples"]
        self._dim_latent = _config["dim_latent"]

        self._metric_names = ["mse", "logprob_VI", "logprob_ML", "logprob_ML_IW"]

    def __call__(self, data, is_training=tf.constant(True, dtype=tf.bool), epoch=tf.constant(0, dtype=tf.int32)):
        """
        data: dictionary-like form, with attributes "tr_input", "val_input", "tr_output", "val_output"
            Classification training/validation data of a task.
        is_training: bool
            If True, training mode
        """
        # Initialise Variables #TODO: is_training set as tf.constant in learner
        self.is_training.assign(is_training)
        self.epoch.assign(epoch)
        assert self.has_initialised, "the learner is not initialised, please initialise via the method - .initialise"

        tr_input, val_input, tr_output, val_output = data.tr_input, data.val_input, data.tr_output, data.val_output
        all_input = tf.concat([tr_input, val_input], axis=-2)
        all_output = tf.concat([tr_output, val_output], axis=-2)

        #use sum(tr_reprs) and sum(tr_reprs_all) for z distribution, val_reprs for prediction, tr_input_ff from deterministic path as feature transformation
        tr_reprs, val_reprs, tr_input_ff, val_input_ff = self.Encoder(tr_input=tr_input, val_input=val_input, tr_output=tr_output)
        all_reprs, _ = self.Encoder_train_only(tr_input=all_input, tr_output=all_output) #use all datapoints as context
        return self.Decoder(data=data, tr_reprs=tr_reprs, val_reprs=val_reprs, all_reprs=all_reprs, tr_input=tr_input_ff, val_input=val_input_ff, epoch=epoch)

    def Encoder_train_only(self, tr_input, tr_output):

        # Initialise r
        tr_reprs = self.forward_initialiser(tr_input, is_training=self.is_training)
        
        # Fourier features
        tr_input_ff = self.fourier_features(
            X=tr_input, 
            recompute=tf.math.logical_not(self._indp_iter), 
            precomputed=self.sample_tr_data,
            iteration=0)

        # Precompute target context interaction for kernel/attention
        precomputed = self.forward_kernel_or_attention_precompute(
            querys=tr_input_ff, 
            keys=tr_input_ff,
            recompute=tf.math.logical_not(self._indp_iter), # if not indepnedent iteration, precompute 
            precomputed=self.precomputed_init,
            values=None,
            iteration=0,
            is_training=self.is_training)

        # Iterative functional updating
        for k in range(self._num_iters):
            updates = self.forward_local_updater(r=tr_reprs, y=tr_output, x=tr_input, iteration=k)

            # Fourier features
            tr_input_ff = self.fourier_features(
                X=tr_input, 
                recompute=self._indp_iter, 
                precomputed=tr_input_ff,
                iteration=k)

            precomputed = self.forward_kernel_or_attention_precompute(
                querys=tr_input_ff, 
                keys=tr_input_ff,
                recompute=self._indp_iter,
                precomputed=precomputed,
                values=None,
                iteration=k,
                is_training=self.is_training)

            tr_updates = self.alpha * self.forward_kernel_or_attention(
                querys=None,
                keys=None,
                precomputed=precomputed,
                values=updates)

            tr_reprs += tr_updates

        return tr_reprs, tr_input_ff

    def Decoder(self, data, tr_reprs, val_reprs, all_reprs, tr_input, val_input, epoch=tf.constant(1, dtype=tf.int32)):
        q_c = self.sample_latent(tr_reprs)
        q_t = self.sample_latent(all_reprs)
        z_samples_c = q_c.sample(sample_shape=self._num_z_samples) #(num_z_samples, batch_size, 1, dim_latent)
        z_samples_t = q_t.sample(sample_shape=self._num_z_samples) #(num_z_samples, batch_size, 1, dim_latent)

        tr_input = tf.repeat(tf.expand_dims(tr_input, axis=0), axis=0, repeats=self._num_z_samples) #for context loss
        val_input = tf.repeat(tf.expand_dims(val_input, axis=0), axis=0, repeats=self._num_z_samples) #for target loss
        tr_output = tf.repeat(tf.expand_dims(data.tr_output, axis=0), axis=0, repeats=self._num_z_samples)
        val_output = tf.repeat(tf.expand_dims(data.val_output, axis=0), axis=0, repeats=self._num_z_samples)

        tr_reprs_c = self.latent_decoder(R=tr_reprs, z_samples=z_samples_c) #(num_z_samples, batch_size, num_target, dim_reprs) #for train prediction
        weights = self.forward_decoder(tr_reprs_c) #(num_z_samples, batch_size, num_target, dim_weights)
        tr_mu, tr_sigma = self.predict(inputs=tr_input, weights=weights)

        tr_reprs_t = self.latent_decoder(R=tr_reprs, z_samples=z_samples_t) #(num_z_samples, batch_size, num_target, dim_reprs) #for posterior sampling loss
        weights = self.forward_decoder(tr_reprs_t) #(num_z_samples, batch_size, num_target, dim_weights)
        tr_mu_pos, tr_sigma_pos = self.predict(inputs=tr_input, weights=weights)

        val_reprs_c = self.latent_decoder(R=val_reprs, z_samples=z_samples_c) #for val prediction
        weights = self.forward_decoder(val_reprs_c)
        val_mu, val_sigma = self.predict(inputs=val_input, weights=weights)

        val_reprs_t = self.latent_decoder(R=val_reprs, z_samples=z_samples_t) #for posterior sampling loss
        weights = self.forward_decoder(val_reprs_t)
        val_mu_pos, val_sigma_pos = self.predict(inputs=val_input, weights=weights)

        if epoch <= self._fixed_sigma_epoch:
            tr_sigma = self._fixed_sigma_value
            tr_sigma_pos = self._fixed_sigma_value
            val_sigma = self._fixed_sigma_value
            val_sigma_pos = self._fixed_sigma_value
        else:
            tr_sigma = tr_sigma
            tr_sigma_pos = tr_sigma_pos
            val_sigma = val_sigma
            val_sigma_pos = val_sigma_pos

        tr_loss, tr_metric = self.calculate_loss_and_metrics(target_y=tr_output, mus_c=tr_mu, sigmas_c=tr_sigma, mus_t=tr_mu_pos, sigmas_t=tr_sigma_pos, q_c=q_c, q_t=q_t, z_samples_c=z_samples_c, z_samples_t=z_samples_t)
        val_loss, val_metric = self.calculate_loss_and_metrics(target_y=val_output, mus_c=val_mu, sigmas_c=val_sigma, mus_t=val_mu_pos, sigmas_t=val_sigma_pos, q_c=q_c, q_t=q_t, z_samples_c=z_samples_c, z_samples_t=z_samples_t)

        additional_loss = tf.constant(0., dtype=self._float_dtype)

        additional_loss = tf.constant(0., dtype=self._float_dtype)
        return val_loss, additional_loss, tr_metric, val_metric, val_mu, val_sigma, tr_mu, tr_sigma

    @snt.once
    def sample_latent_init(self):
        self._sample_latent = submodules.sample_latent(
            nn_layers=self._nn_layers,
            nn_size=self._nn_size,
            dim_latent=self._dim_latent,
            stddev_const_scale=self._stddev_const_scale,
            initialiser=self.initialiser,
            nonlinearity=self._nonlinearity,
            stddev_offset=self._stddev_offset)

    def sample_latent(self, reprs):
        return self._sample_latent(reprs=reprs)

    @snt.once
    def latent_decoder_init(self):
        self._latent_decoder = submodules.latent_decoder(
            initialiser=self.initialiser,
            dim_reprs=self._dim_reprs,
            nonlinearity=self._nonlinearity,
            no_decoder=self._no_decoder)

    def latent_decoder(self, R, z_samples):
        return self._latent_decoder(R=R, z_samples=z_samples)

    @snt.once
    def calculate_loss_and_metrics_init(self):

        def mse_loss(target_y, mus_c, mus_t, sigmas_c, sigmas_t, q_c=None, q_t=None, z_samples_c=None, z_samples_t=None):
            mus, sigmas, z_samples = mus_c, sigmas_c, z_samples_c
            mu, sigma = mus, sigmas
            mse = self.loss_fn(mu, target_y)
            return mse

        def log_prob_VI_loss(target_y, mus_c, mus_t, sigmas_c, sigmas_t, q_c, q_t, z_samples_c=None, z_samples_t=None):
            mus, sigmas, z_samples = mus_t, sigmas_t, z_samples_t
            p_y = tfp.distributions.MultivariateNormalDiag(loc=mus, scale_diag=sigmas)
            E_p_y = tf.math.reduce_sum(tf.math.reduce_mean(p_y.log_prob(target_y), axis=0),axis=-1) #(batch_size)
            kl = tf.squeeze(tfp.distributions.kl_divergence(q_t, q_c)) #(batch_size)
            loss = - (E_p_y - kl)
            return loss

        def log_prob_ML_loss(target_y, mus_c, mus_t, sigmas_c, sigmas_t, z_samples_c, z_samples_t, q_c, q_t):
            mus, sigmas, z_samples = mus_c, sigmas_c, z_samples_c
            num_z_samples = tf.shape(z_samples)[0]
            p_y = tfp.distributions.MultivariateNormalDiag(loc=mus, scale_diag=sigmas)
            loss = tf.math.reduce_sum(p_y.log_prob(target_y),axis=-1) #(num_z_samples, batch_size)
            loss = tf.math.reduce_logsumexp(loss, axis=0) #(batch_size)
            loss = - (loss - tf.math.log(tf.cast(num_z_samples, dtype=tf.float32)))
            return loss

        def log_prob_ML_IW_loss(target_y, mus_c, sigmas_c, mus_t, sigmas_t, z_samples_c, z_samples_t, q_c, q_t):
            mus, sigmas, z_samples = mus_t, sigmas_t, z_samples_t
            num_z_samples = tf.shape(z_samples)[0]
            p_y = tfp.distributions.MultivariateNormalDiag(loc=mus, scale_diag=sigmas)
            sum_p_y = tf.math.reduce_sum(p_y.log_prob(target_y),axis=-1) #(num_z_samples, batch_size)
            loss = sum_p_y + tf.squeeze(q_c.log_prob(z_samples)) - tf.squeeze(q_t.log_prob(z_samples)) #importance sampling
            loss = tf.math.reduce_logsumexp(loss, axis=0) #(batch_size)
            loss = - (loss - tf.math.log(tf.cast(num_z_samples, dtype=tf.float32)))
            return loss

        def loss_and_metric(loss_fn, target_y, mus_c, sigmas_c, mus_t, sigmas_t, q_c, q_t, z_samples_c=None, z_samples_t=None):
            n_points = tf.cast(tf.shape(target_y)[-2], dtype=tf.float32)

            loss = loss_fn(target_y=target_y, mus_c=mus_c, sigmas_c=sigmas_c, mus_t=mus_t, sigmas_t=sigmas_t, q_c=q_c, q_t=q_t, z_samples_c=z_samples_c, z_samples_t=z_samples_t)
            mse = mse_loss(target_y=target_y, mus_c=mus_c, sigmas_c=sigmas_c, mus_t=mus_t, sigmas_t=sigmas_t, q_c=q_c, q_t=q_t, z_samples_c=z_samples_c, z_samples_t=z_samples_t)
            logprob_VI = tf.math.divide_no_nan(- log_prob_VI_loss(target_y=target_y, mus_c=mus_c, sigmas_c=sigmas_c, mus_t=mus_t, sigmas_t=sigmas_t, q_c=q_c, q_t=q_t, z_samples_c=z_samples_c, z_samples_t=z_samples_t), n_points)
            logprob_ML = tf.math.divide_no_nan(- log_prob_ML_loss(target_y=target_y, mus_c=mus_c, sigmas_c=sigmas_c, mus_t=mus_t, sigmas_t=sigmas_t, q_c=q_c, q_t=q_t, z_samples_c=z_samples_c, z_samples_t=z_samples_t), n_points)
            logprob_ML_IW = tf.math.divide_no_nan(- log_prob_ML_IW_loss(target_y=target_y, mus_c=mus_c, sigmas_c=sigmas_c, mus_t=mus_t, sigmas_t=sigmas_t, q_c=q_c, q_t=q_t, z_samples_c=z_samples_c, z_samples_t=z_samples_t), n_points)
            # if tf.shape(target_y)[-2] != tf.constant(0, dtype=tf.int32): # to avoid nan when computing metrics
            #     logprob_VI = - log_prob_VI_loss(target_y=target_y, mus=mus, sigmas=sigmas, q_c=q_c, q_t=q_t, z_samples=z_samples) / n_points
            #     logprob_ML = - log_prob_ML_loss(target_y=target_y, mus=mus, sigmas=sigmas, q_c=q_c, q_t=q_t, z_samples=z_samples) / n_points
            # else:
            #     logprob_VI = - log_prob_VI_loss(target_y=target_y, mus=mus, sigmas=sigmas, q_c=q_c, q_t=q_t, z_samples=z_samples) / n_points
            #     logprob_ML = - log_prob_ML_loss(target_y=target_y, mus=mus, sigmas=sigmas, q_c=q_c, q_t=q_t, z_samples=z_samples) / n_points
                # logprob_VI = tf.zeros(tf.shape(mus)[:2])[0,:] #empty shape
                # logprob_ML = tf.zeros(tf.shape(mus)[:2])[0,:] #empty shape
            return loss, [mse, logprob_VI, logprob_ML, logprob_ML_IW]

        if self._loss_type == "mse":
            self._calculate_loss_and_metrics =  partial(loss_and_metric, loss_fn=mse_loss)
        elif self._loss_type == "logprob_VI" or self._loss_type == "logprob":
            self._calculate_loss_and_metrics =  partial(loss_and_metric, loss_fn=log_prob_VI_loss)
        elif self._loss_type == "logprob_ML":
            self._calculate_loss_and_metrics =  partial(loss_and_metric, loss_fn=log_prob_ML_loss)
        elif self._loss_type == "logprob_ML_IW":
            self._calculate_loss_and_metrics =  partial(loss_and_metric, loss_fn=log_prob_ML_IW_loss)
        else:
            raise NameError("unknown loss type")

    def calculate_loss_and_metrics(self, target_y, mus_c, sigmas_c, mus_t, sigmas_t, q_c, q_t, z_samples_c, z_samples_t):
        return self._calculate_loss_and_metrics(target_y=target_y, mus_c=mus_c, sigmas_c=sigmas_c, mus_t=mus_t, sigmas_t=sigmas_t, q_c=q_c, q_t=q_t, z_samples_c=z_samples_c, z_samples_t=z_samples_t)
        

class MetaFunRegressorV4(MetaFunRegressorV3):
    def __init__(self, config, data_source="regression", no_batch=False, name="MetaFunRegressorV4", **kwargs):
        super(MetaFunRegressorV4, self).__init__(config=config, data_source=data_source, no_batch=no_batch, name=name)

    @snt.once
    def predict_init(self):
        """ backend of decoder to produce mean and variance of predictions"""
        def predict_no_decoder1(inputs, weights):
            return weights, tf.ones_like(weights) * 0.5

        def predict_no_decoder2(inputs, weights):
            return tf.split(weights, 2, axis=-1)

        def predict_repr_as_inputs(inputs, weights):
            preds_input = tf.broadcast_to(self.pseudo_input, shape=tf.concat([tf.shape(inputs)[:-1], tf.constant([self._decoder_output_sizes[0]], dtype=tf.int32)], axis=0))
            preds = self._predict_repr_as_inputs(inputs=preds_input, weights=weights)
            return _split(preds)

        def predict_repr_as_inputs_simple(inputs, weights):
            preds = self._predict_repr_as_inputs_simple(inputs=inputs, weights=weights)
            return _split(preds)

        def predict_not_repr_as_inputs(inputs, weights):
            preds_input = tf.broadcast_to(self.pseudo_input, shape=tf.concat([tf.shape(inputs)[:-1], tf.constant([self._decoder_output_sizes[0]], dtype=tf.int32)], axis=0))
            preds = self.custom_MLP(inputs=preds_input, weights=weights)
            return _split(preds)
        
        def _split(preds):
            mu, log_sigma = tf.split(preds, 2, axis=-1)
            sigma = self._stddev_const_scale + (1-self._stddev_const_scale) * tf.nn.softplus(log_sigma-self._stddev_offset)
            return mu, sigma

        if self._no_decoder:
            if self._dim_reprs == 1:
                self._predict = predict_no_decoder1
            elif self._dim_reprs == 2:
                self._predict = predict_no_decoder2
            else:
                raise Exception("num_reprs must <=2 if no_decoder")
        elif self._repr_as_inputs == "simple":
                self._predict_repr_as_inputs_simple = submodules.predict_repr_as_inputs_simple(
                    nn_size=self._nn_size,
                    nn_layers=self._nn_layers, 
                    output_dim=self.output_dim,
                    initialiser=self.initialiser,
                    nonlinearity=self._nonlinearity
                )
                self._predict =  predict_repr_as_inputs_simple

        else:
            self.pseudo_input = tf.Variable(
                    tf.constant_initializer(0.)(
                        shape=[self._decoder_output_sizes[0]],
                        dtype = self._float_dtype),
                    trainable=True,
                    name="predict_pseudo_input")
            
            if self._repr_as_inputs is True:
                self._predict_repr_as_inputs = submodules.predict_repr_as_inputs(
                    output_sizes=self._decoder_output_sizes[1:],
                    initialiser=self.initialiser,
                    nonlinearity=self._nonlinearity
                )
                self._predict =  predict_repr_as_inputs
            else:
                self.custom_MLP = submodules.custom_MLP(
                    output_sizes=self._decoder_output_sizes[1:],
                    embedding_dim=self._decoder_output_sizes[0],
                    nonlinearity=self._nonlinearity)
                self._predict =  predict_not_repr_as_inputs


class MetaFunRegressorGLV4(MetaFunRegressorGLV3, MetaFunRegressorV4):
    def __init__(self, config, data_source="regression", no_batch=False, name="MetaFunRegressorGLV4", **kwargs):
        super(MetaFunRegressorGLV4, self).__init__(config=config, data_source=data_source, no_batch=no_batch, name=name)


class MetaFunBaseGLV3(MetaFunBaseV2):
    def __init__(self, config, no_batch, num_classes=1, name="MetaFunGLV3", **kwargs):
        super(MetaFunBaseGLV3, self).__init__(config=config, no_batch=no_batch, num_classes=num_classes, name=name)

    @snt.once
    def additional_initialise_after(self, data_instance=None):
        # Deduce precompute shape of forward_kernel_or_attention_precompute
        self.precomputed_init = tf.zeros_like(self.forward_kernel_or_attention_precompute(
            querys=tf.concat([self.sample_tr_data, self.sample_val_data], axis=-2),
            keys=self.sample_tr_data,
            values=None,
            recompute=True,
            precomputed=None,
            iteration=0,
            is_training=self.is_training
        ))

        self.deterministic_encoder_cls.initialise(data_instance=data_instance)
        self.sample_latent_init()


class MetaFunRegressorGLV5(MetaFunBaseGLV3, MetaFunRegressorGLV4):
    def __init__(self, config, data_source="regression", no_batch=False, name="MetaFunRegressorGLV5", **kwargs):
        super(MetaFunRegressorGLV5, self).__init__(config=config, data_source=data_source, no_batch=no_batch, name=name)

        self.deterministic_encoder_cls = MetaFunRegressorV4(config=config, data_source=data_source, no_batch=no_batch, name="deterministic_encoder")
        utils.disable_decoder(self.deterministic_encoder_cls)

    def __call__(self, data, is_training=tf.constant(True, dtype=tf.bool), epoch=tf.constant(0, dtype=tf.int32)):
        """
        data: dictionary-like form, with attributes "tr_input", "val_input", "tr_output", "val_output"
            Classification training/validation data of a task.
        is_training: bool
            If True, training mode
        """
        # Initialise Variables #TODO: is_training set as tf.constant in learner
        self.is_training.assign(is_training)
        self.epoch.assign(epoch)
        self.deterministic_encoder_cls.is_training.assign(is_training)
        self.deterministic_encoder_cls.epoch.assign(epoch)
        assert self.has_initialised, "the learner is not initialised, please initialise via the method - .initialise"
    
        tr_input, val_input, tr_output, val_output = data.tr_input, data.val_input, data.tr_output, data.val_output
        all_input = tf.concat([tr_input, val_input], axis=-2)
        all_output = tf.concat([tr_output, val_output], axis=-2)

        tr_reprs_deter, val_reprs_deter, tr_input_ff, val_input_ff = self.deterministic_encoder(tr_input=tr_input, val_input=val_input, tr_output=tr_output) #deterministic
        tr_reprs_latent, val_reprs_latent, _, _ = super().Encoder(tr_input=tr_input, val_input=val_input, tr_output=tr_output) # latent encoder
        all_reprs_latent, _ = super().Encoder_train_only(tr_input=all_input, tr_output=all_output) # use all datapoints as conte

        return self.Decoder(data=data, tr_reprs_deter=tr_reprs_deter, val_reprs_deter=val_reprs_deter, tr_reprs_latent=tr_reprs_latent, val_reprs_latent=val_reprs_latent, all_reprs_latent=all_reprs_latent, tr_input=tr_input_ff, val_input=val_input_ff, epoch=epoch)

    def deterministic_encoder(self, tr_input, val_input, tr_output):
        return self.deterministic_encoder_cls.Encoder(tr_input=tr_input, val_input=val_input, tr_output=tr_output)

    @snt.once
    def deterministic_decoder_init(self):
        self._deterministic_decoder = submodules.simple_MLP(
            nn_size=self._nn_size,
            nn_layers=self._nn_layers, 
            output_dim=self._dim_reprs,
            initialiser=self.initialiser,
            nonlinearity=self._nonlinearity
            )
    
    def deterministic_decoder(self, reprs):
        return self.deterministic_decoder(inputs=reprs)

    def Decoder(self, data, tr_reprs_deter, val_reprs_deter, tr_reprs_latent, val_reprs_latent, all_reprs_latent, tr_input, val_input, epoch=tf.constant(1, dtype=tf.int32)):
        q_c = self.sample_latent(tr_reprs_latent)
        q_t = self.sample_latent(all_reprs_latent)
        z_samples_c = q_c.sample(sample_shape=self._num_z_samples) #(num_z_samples, batch_size, 1, dim_latent)
        z_samples_t = q_t.sample(sample_shape=self._num_z_samples) #(num_z_samples, batch_size, 1, dim_latent)

        tr_input = tf.repeat(tf.expand_dims(tr_input, axis=0), axis=0, repeats=self._num_z_samples) #for context loss
        val_input = tf.repeat(tf.expand_dims(val_input, axis=0), axis=0, repeats=self._num_z_samples) #for target loss
        tr_output = tf.repeat(tf.expand_dims(data.tr_output, axis=0), axis=0, repeats=self._num_z_samples)
        val_output = tf.repeat(tf.expand_dims(data.val_output, axis=0), axis=0, repeats=self._num_z_samples)

        tr_reprs_c = submodules.latent_deter_merger(R=tr_reprs_deter, z_samples=z_samples_c) #(num_z_samples, batch_size, num_target, dim_reprs) #for train prediction
        weights = self.forward_decoder(tr_reprs_c) #(num_z_samples, batch_size, num_target, dim_weights)
        tr_mu, tr_sigma = self.predict(inputs=tr_input, weights=weights)

        tr_reprs_t = submodules.latent_deter_merger(R=tr_reprs_deter, z_samples=z_samples_t) #(num_z_samples, batch_size, num_target, dim_reprs) #for posterior sampling loss
        weights = self.forward_decoder(tr_reprs_t) #(num_z_samples, batch_size, num_target, dim_weights)
        tr_mu_pos, tr_sigma_pos = self.predict(inputs=tr_input, weights=weights)

        val_reprs_c = submodules.latent_deter_merger(R=val_reprs_deter, z_samples=z_samples_c) #for val prediction
        weights = self.forward_decoder(val_reprs_c)
        val_mu, val_sigma = self.predict(inputs=val_input, weights=weights)

        val_reprs_t = submodules.latent_deter_merger(R=val_reprs_deter, z_samples=z_samples_t) #for posterior sampling loss
        weights = self.forward_decoder(val_reprs_t)
        val_mu_pos, val_sigma_pos = self.predict(inputs=val_input, weights=weights)

        if epoch <= self._fixed_sigma_epoch:
            tr_sigma = self._fixed_sigma_value
            tr_sigma_pos = self._fixed_sigma_value
            val_sigma = self._fixed_sigma_value
            val_sigma_pos = self._fixed_sigma_value
        else:
            tr_sigma = tr_sigma
            tr_sigma_pos = tr_sigma_pos
            val_sigma = val_sigma
            val_sigma_pos = val_sigma_pos

        tr_loss, tr_metric = self.calculate_loss_and_metrics(target_y=tr_output, mus_c=tr_mu, sigmas_c=tr_sigma, mus_t=tr_mu_pos, sigmas_t=tr_sigma_pos, q_c=q_c, q_t=q_t, z_samples_c=z_samples_c, z_samples_t=z_samples_t)
        val_loss, val_metric = self.calculate_loss_and_metrics(target_y=val_output, mus_c=val_mu, sigmas_c=val_sigma, mus_t=val_mu_pos, sigmas_t=val_sigma_pos, q_c=q_c, q_t=q_t, z_samples_c=z_samples_c, z_samples_t=z_samples_t)

        additional_loss = tf.constant(0., dtype=self._float_dtype)

        return val_loss, additional_loss, tr_metric, val_metric, val_mu, val_sigma, tr_mu, tr_sigma

        


if __name__ == "__main__":
    from utils import parse_config
    import os
    import numpy as np
    import collections
    config = parse_config(os.path.join(os.path.dirname(__file__),"config/debug.yaml"))
    #tf.random.set_seed(1234)
    #np.random.seed(1234)

    # ClassificationDescription = collections.namedtuple(
    # "ClassificationDescription",
    # ["tr_input", "tr_output", "val_input", "val_output"])
    
    # data = ClassificationDescription(
    # tf.constant(np.random.random([10,10]),dtype=tf.float32),
    # tf.constant(np.random.uniform(1,10,10).reshape(-1,1),dtype=tf.int32),
    # tf.constant(np.random.random([10,10]),dtype=tf.float32),
    # tf.constant(np.random.uniform(1,10,10).reshape(-1,1),dtype=tf.int32))

    # print(module(data))


    # ####V1
    # print("Classification")
    # from data.leo_imagenet import DataProvider
    # module = MetaFunClassifier(config=config)
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

    # ####V2
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

    # ###V3
    # import time

    # print("Regression")
    # module2 = MetaFunRegressorV3(config=config)
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
 
    # #print(trial(data_reg, is_training=False))
    # t1 = time.time()
    # print(module2(data_reg, is_training=False)[0])
    # print("time consumed", time.time()-t1)

    # ####GLV3
    # import time

    # print("Regression")
    # module2 = MetaFunRegressorGLV3(config=config)
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
    # print("-----tffunction")
    # print(trial(data_reg, is_training=False))

    # t1 = time.time()
    # print("-----normal")
    # print(module2(data_reg, is_training=False)[0])
    # print(module2(data_reg, is_training=False)[3][1:])
    # print("time consumed", time.time()-t1)


    # ####V4
    # import time
    # config = parse_config(os.path.join(os.path.dirname(__file__),"config/config22.yaml"))

    # print("Regression")
    # module2 = MetaFunRegressorV4(config=config)
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
    # t1 = time.time()
    # print(module2(data_reg, is_training=False)[0])
    # print("time consumed", time.time()-t1)

    ####GLV5
    import time

    print("Regression")
    module2 = MetaFunRegressorGLV5(config=config)
    ClassificationDescription = collections.namedtuple(
    "ClassificationDescription",
    ["tr_input", "tr_output", "val_input", "val_output"])
    
    data_reg = ClassificationDescription(
    tf.constant(np.random.random([2, 10,10]),dtype=tf.float32),
    tf.constant(np.random.random([2,10,1]),dtype=tf.float32),
    tf.constant(np.random.random([2, 10,10]),dtype=tf.float32),
    tf.constant(np.random.random([2,10,1]),dtype=tf.float32))

    module2.initialise(data_reg)
    data_reg = ClassificationDescription(
    tf.constant(np.random.random([2, 10,10]),dtype=tf.float32),
    tf.constant(np.random.random([2,10,1]),dtype=tf.float32),
    tf.constant(np.random.random([2, 10,10]),dtype=tf.float32),
    tf.constant(np.random.random([2,10,1]),dtype=tf.float32))
    @tf.function
    def trial(x, is_training=True):
        l,*_ = module2(x, is_training=is_training)
        return l

    print("DEBUGGGGGGGGGGGGGGG")
    print("-----tffunction")
    #print(trial(data_reg, is_training=False))

    t1 = time.time()
    print("-----normal")
    print(module2(data_reg, is_training=False)[3])
    print("time consumed", time.time()-t1)


    ######### invariant check
    a1 = tf.constant(np.random.random([2, 10,10]),dtype=tf.float32)
    a2 = tf.constant(np.random.random([2,10,1]),dtype=tf.float32)
    a3 = tf.constant(np.random.random([2, 10,10]),dtype=tf.float32)
    a4 = tf.constant(np.random.random([2,10,1]),dtype=tf.float32)
    data_reg = ClassificationDescription(a1,a2,a3,a4)
    _,_,_,_,b1,*_ = module2(data_reg)

    c1, c2, c3, c4 = a1+10000, a2, a3+10000, a4
    data_reg2 = ClassificationDescription(c1,c2,c3,c4)
    _,_,_,_,b2,*_ = module2(data_reg2)

    print(b1[0]-b2[0])
