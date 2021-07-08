import tensorflow as tf
import sonnet as snt


#################################################################################
# Initialisers
#################################################################################
class constant_initialiser(snt.Module):
    """constant initialiser"""
    def __init__(self, dim_reprs, float_dtype, classification=True, num_classes=1, no_batch=False, trainable=False, name="constant_initialiser"):
        super(constant_initialiser, self).__init__(name=name)
        if trainable:
            self.init = tf.Variable(
                    tf.constant_initializer(0.)(
                        shape=[1, dim_reprs],
                        dtype = float_dtype),
                        trainable=True,
                        name="initial_state")
        else:
            self.init = tf.zeros([1, dim_reprs])

        self._num_classes = num_classes if classification else 1

        if no_batch:
            self.tile_fun = lambda init, x : tf.tile(init, [tf.shape(x)[-2], self._num_classes])
        else:
            def tile_fun_batch(init, x):
                return tf.tile(tf.expand_dims(init,0),[tf.shape(x)[-3], tf.shape(x)[-2], self._num_classes])

            self.tile_fun = tile_fun_batch

    def __call__(self, x):
        return self.tile_fun(self.init, x)

class parametric_initialiser(snt.Module):
    """parametric initialiser"""
    def __init__(self, nn_size, nn_layers, dim_reprs, dropout_rate, initialiser, nonlinearity, classification=True, num_classes=1, name="parametric_initialiser"):
        super(parametric_initialiser, self).__init__(name=name)
        self._num_classes = num_classes if classification else 1
        self.module = snt.nets.MLP( # dtype depends on input #TODO: cast input
            output_sizes=[nn_size] * nn_layers + [dim_reprs],
            w_init = initialiser,
            with_bias=True,
            activation=nonlinearity,
            name="parametric_initialiser"
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def __call__(self, x, is_training=True):
        after_dropout = self.dropout(x, training=is_training)
        outputs = self.module(after_dropout)
        return tf.concat([outputs for i in range(self._num_classes)],axis=-1)


#################################################################################
# Forward local updater
#################################################################################
class neural_local_updater(snt.Module):
    """neural local updater"""
    def __init__(self, nn_size, nn_layers, dim_reprs, initialiser, nonlinearity, classification=True, num_classes=1, no_batch=False, xNone=False, name="neural_local_updater"):
        super(neural_local_updater, self).__init__(name=name)
        if classification:
            self._num_classes = num_classes
            self._dim_reprs = dim_reprs

            # MLP m
            self.module1 = snt.nets.MLP(
                output_sizes=[nn_size] * nn_layers,
                w_init=initialiser,
                with_bias=True,
                activation=nonlinearity,
                name="neural_local_updater_m"
            )

            # MLP u+
            self.module2 = snt.nets.MLP(
                output_sizes=[nn_size] * nn_layers + [dim_reprs],
                w_init=initialiser,
                with_bias=True,
                activation=nonlinearity,
                name="neural_local_updater_uplus"
            )

            # MLP u-
            self.module3 = snt.nets.MLP(
                output_sizes=[nn_size] * nn_layers + [dim_reprs],
                w_init=initialiser,
                with_bias=True,
                activation=nonlinearity,
                name="neural_local_updater_uminus"
            )

            if no_batch:
                self.perm = [0, 2, 1]
                self.tile_dim = [1, self._num_classes, 1]
            else:
                self.perm = [0, 1, 3, 2]
                self.tile_dim = [1, 1, self._num_classes, 1]
            
            self.call_fn = self.classification

        else:
            if not xNone:
                self.concat_fn = lambda r, y, x: tf.concat([r, y, x], axis=-1)
            else:
                self.concat_fn = lambda r, y, x: tf.concat([r, y], axis=-1)
            self.module = snt.nets.MLP(
                output_sizes=[nn_size] * nn_layers + [dim_reprs],
                w_init=initialiser,
                with_bias=True,
                activation=nonlinearity,
                name="neural_local_updater"
            )

            self.call_fn = self.regression

        

    def classification(self, r, y, x=None, iter=""):
        r_shape = tf.shape(r)
        r = tf.reshape(r, tf.concat([r_shape[:-1], tf.constant([self._num_classes, self._dim_reprs], dtype=tf.int32)], axis=0))

        y = tf.one_hot(y, self._num_classes) #TODO: move to data preprocessing
        y = tf.transpose(y, self.perm)
        outputs = self.module1(r)
        agg_outputs = tf.math.reduce_mean(outputs, axis=-2, keepdims=True)
        outputs = tf.concat([outputs, tf.tile(agg_outputs, self.tile_dim)], axis=-1)

        outputs_t = self.module2(outputs)

        outputs_f = self.module3(outputs)
        outputs = outputs_t * y + outputs_f * (1-y)
        outputs = tf.reshape(outputs, shape=r_shape)
        return outputs
    
    def regression(self, r, y, x, iter=""):
        reprs = self.concat_fn(r, y, x)
        return self.module(reprs)

    def __call__(self, r, y, x=None, iter=""):
        return self.call_fn(r=r, y=y, x=x, iter=iter)


#################################################################################
# Decoder
#################################################################################
class decoder(snt.Module):
    """decoder"""
    def __init__(self, embedding_dim, initialiser, classification=True, nn_size=1, nn_layers=1, nonlinearity=tf.nn.relu, orthogonality_penalty_weight=0, repr_as_inputs=False, regression_output_sizes=[40,40,2], name="decoder"):
        super(decoder, self).__init__(name=name)

        self._repr_as_inputs = repr_as_inputs
        if classification:
            self.orthogonality_reg = get_orthogonality_regularizer(orthogonality_penalty_weight)
            self.decoder_module = snt.Linear(
                output_size=embedding_dim * 2,
                with_bias=True,
                w_init=initialiser
            )
        elif not repr_as_inputs:
            num_layers = len(regression_output_sizes)
            output_sizes = [embedding_dim] + regression_output_sizes
            # Count number of parameters in the predictor
            num_params = 0
            for i in range(num_layers):
                num_params += (output_sizes[i]+1) * output_sizes[i+1]
            self.decoder_module = snt.nets.MLP(
                output_sizes=[nn_size] * nn_layers + [2 * num_params],
                activation=nonlinearity,
                with_bias=True,
                w_init=initialiser,
                name="decoder"
            )


        self.call_fn = self.classification if classification else self.regression

    def __call__(self, inputs):
        return self.call_fn(inputs)

    def classification(self, inputs):
        outputs = self.decoder_module(inputs)
        return outputs, self.orthogonality_reg(self.decoder_module.w)

    def regression(self, reprs):
        if self._repr_as_inputs:
            return reprs
        else:
            return self.decoder_module(reprs)


# (Adapted from https://github.com/deepmind/leo, see copyright and original license in our LICENSE file.)
def get_orthogonality_regularizer(orthogonality_penalty_weight):
    """Returns the orthogonality regularizer."""
    def orthogonality(weight):
        """Calculates the layer-wise penalty encouraging orthogonality."""
        w2 = tf.linalg.matmul(weight, weight, transpose_b=True)
        wn = tf.linalg.norm(weight, ord=2, axis=1, keepdims=True) + 1e-32
        correlation_matrix = w2 / tf.matmul(wn, wn, transpose_b=True)
        matrix_size = tf.shape(correlation_matrix)[0]
        base_dtype = weight.dtype.base_dtype
        identity = tf.linalg.eye(matrix_size,dtype=base_dtype)
        weight_corr = tf.reduce_mean(
            tf.math.squared_difference(correlation_matrix, identity))
        return tf.multiply(
            tf.cast(orthogonality_penalty_weight, base_dtype),
            weight_corr,
            name="orthogonality_regularisation"
        )
    return orthogonality



#################################################################################
# Attention and kernel
#################################################################################

#### Attention
def dot_product_attention_frontend_fn(querys, keys, normalise):
    """ dot product attention frontend for precomputation"""
    d_k = tf.shape(querys)[-1]
    scale = tf.math.sqrt(tf.cast(d_k, tf.float32))
    unnorm_weights = tf.linalg.matmul(querys, keys, transpose_b=True) / scale # [B,m,n]
    if normalise:
        weights = tf.math.softmax(unnorm_weights)
    else:
        weights = tf.math.sigmoid(unnorm_weights)
    return weights

def dot_product_attention_backend_fn(weights, values):
    """dot product attention backend which takes output from frontend"""
    return tf.linalg.matmul(weights, values)

def dot_product_attention_fn(querys, keys, values, normalise):
    """Computes dot product attention.

    Args:
        querys: queries. tensor of  shape [B,m,d_k].
        keys: keys. tensor of shape [B,n,d_k].
        vaues: values. tensor of shape [B,n,d_v].
        normalise: Boolean that determines whether weights sum to 1.
    Returns:
        tensor of shape [B,m,d_v].
    """
    weights = dot_product_attention_frontend_fn(querys=querys, keys=keys, normalise=normalise)
    return dot_product_attention_backend_fn(weights=weights, values=values)

class Attention(snt.Module):
    """attention - return frontend or backend results"""
  
    def __init__(self, config, complete_return=True, name="attention"):
        """
        config: dict
            - configuration
        complete_return: bool
            - If True, return backend result, else only frontend result
        """
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
        self.initialiser = tf.keras.initializers.GlorotUniform()

        if self._rep == "mlp":
            self.module = snt.nets.MLP( # mapping a
                output_sizes=self._output_sizes,
                w_init=self.initialiser,
                with_bias=True,
                activation=self._nonlinearity,
                name="deep_attention"
                )

        if self._att_type != tf.constant("dot_product",tf.string):
            raise NameError("Unknown attention type")

        if self._rep not in [tf.constant("identity", dtype=tf.string), tf.constant("mlp", dtype=tf.string)]:
            raise NameError("Unknown attention representation - not among ['identity', 'mlp']")

        if self._rep == tf.constant("identity", dtype=tf.string):
            self.call_fn_frontend = lambda keys, querys: dot_product_attention_frontend_fn(keys=keys, querys=querys, normalise=self._normalise)
        else:
            self.call_fn_frontend = lambda keys, querys: dot_product_attention_frontend_fn(keys=self.module(keys), querys=self.module(querys), normalise=self._normalise)
        
        if complete_return:
            self.call_fn_backend = lambda weights, values: dot_product_attention_backend_fn(weights=weights, values=values)
        else:
            self.call_fn_backend = lambda weights, values: weights

    def __call__(self, keys, querys, values=None):
        weights = self.call_fn_frontend(keys=keys, querys=querys)
        return self.call_fn_backend(weights=weights, values=values)

    def backend(self, weights, values):
        return dot_product_attention_backend_fn(weights=weights, values=values)

#### Squared-exponential kernel
def squared_exponential_kernel_frontend_fn(querys, keys, sigma, lengthscale):
    """frontend for se kernel"""
    sq_norm = tf.reduce_sum((tf.expand_dims(keys, -3) - tf.expand_dims(querys, -2))**2, axis=-1)
    sq_norm = tf.linalg.matrix_transpose(sq_norm)
    return sigma**2 * tf.math.exp(- sq_norm / (2.*lengthscale**2)) # RBF

def squared_exponential_kernel_backend_fn(query_key, values):
    """backend for se kernel"""
    return tf.linalg.matmul(query_key, values, transpose_a=True) # RBF * V

def squared_exponential_kernel_fun(querys, keys, values, sigma, lengthscale):
    """se (rbf) kernel"""
    kernel_qk = squared_exponential_kernel_frontend_fn(querys=querys, keys=keys, sigma=sigma, lengthscale=lengthscale)
    return squared_exponential_kernel_backend_fn(query_key=kernel_qk, values=values) 

class squared_exponential_kernel(snt.Module):
    def __init__(self, complete_return=True, name="se_kernel"):
        super(squared_exponential_kernel, self).__init__(name=name)

        self.call_fn_frontend = lambda querys, keys, sigma, lengthscale: squared_exponential_kernel_frontend_fn(querys=querys, keys=keys, sigma=sigma, lengthscale=lengthscale)

        if complete_return:
            self.call_fn_backend = lambda query_key, values: squared_exponential_kernel_backend_fn(query_key=query_key, values=values)
        else:
            self.call_fn_backend = lambda query_key, values: query_key

    def __call__(self, querys, keys, sigma, lengthscale, values=None):
        query_key = self.call_fn_frontend(querys=querys, keys=keys, sigma=sigma, lengthscale=lengthscale)
        return self.call_fn_backend(query_key=query_key, values=values)

    def backend(self, query_key, values):
        return squared_exponential_kernel_backend_fn(query_key=query_key, values=values)

#### Deep squared-exponential kernel
class deep_se_kernel(snt.Module): #TODO: clarify whether nn_layer or embedding dim should be used for nerual layer width
    def __init__(self, embedding_layers, kernel_dim, initialiser, nonlinearity, complete_return=True, name="deep_se_kernel"):
        super(deep_se_kernel, self).__init__(name=name)
        self.module = snt.nets.MLP( # mapping a
            output_sizes=[kernel_dim] * embedding_layers,
            w_init=initialiser,
            with_bias=True,
            activation=nonlinearity,
            name="deep_se_kernel"
        )

        self.call_fn_frontend = lambda querys, keys, sigma, lengthscale: squared_exponential_kernel_frontend_fn(
            querys=self.module(querys), keys=self.module(keys), sigma=sigma, lengthscale=lengthscale)

        if complete_return:
            self.call_fn_backend = lambda query_key, values: squared_exponential_kernel_backend_fn(query_key=query_key, values=values)
        else:
            self.call_fn_backend = lambda query_key, values: query_key

    def __call__(self, querys, keys, sigma, lengthscale, values=None):
        query_key = self.call_fn_frontend(querys=querys, keys=keys, sigma=sigma, lengthscale=lengthscale)
        return self.call_fn_backend(query_key=query_key, values=values)

    def backend(self, query_key, values):
        return squared_exponential_kernel_backend_fn(query_key=query_key, values=values)

#################################################################################
# Sampling
#################################################################################

def deterministic_sample(distribution_params, stddev_offset):
    """deterministic sampling by splitting parameteres into mean and stddev directly"""
    means, unnormalised_stddev = tf.split(distribution_params, 2, axis=-1)
    return means

def probabilistic_sample(distribution_params, stddev_offset, is_training=True):
    """probabilistic sampling"""
    means, unnormalized_stddev = tf.split(distribution_params, 2, axis=-1) # mean and log std
    stddev = tf.math.exp(unnormalized_stddev) - (1. - stddev_offset)
    stddev = tf.math.maximum(stddev, 1e-10)
    if not is_training:
        return means
    else:
        return tf.random.normal(shape=means.shape, mean=means, stddev=stddev)


#################################################################################
# Regression predict
#################################################################################

class predict_repr_as_inputs(snt.Module):
    def __init__(self, output_sizes, initialiser, nonlinearity, name="predict"):
        super(predict_repr_as_inputs, self).__init__(name=name)
        self._output_sizes = output_sizes
        self._output_sizes_len = len(output_sizes)
        self._nonlinearity = nonlinearity
        
        self.modules_list = [
            snt.Linear(
                output_size=size,
                with_bias=True,
                w_init=initialiser
            )
            for size in output_sizes
        ]
                    
    def __call__(self, inputs, weights):
        outputs = inputs
        for idx in range(self._output_sizes_len-1):
            outputs = tf.concat([outputs, weights], axis=-1)
            outputs = self.modules_list[idx](outputs)
            outputs = self._nonlinearity(outputs)
        outputs = tf.concat([outputs, weights],axis=-1)
        return self.modules_list[idx+1](outputs)

class custom_MLP(snt.Module):
    def __init__(self, output_sizes, embedding_dim, nonlinearity, name="custom_MLP"):
        super(custom_MLP, self).__init__(name=name)
        output_sizes = [embedding_dim] + output_sizes
        self._output_sizes = output_sizes
        self._num_layers = len(output_sizes)
        self.w_begin = []
        self.w_end = []
        self.b_begin = self.w_end
        self.b_end= []
        self.in_size = self._output_sizes[:-1]
        self.out_size = self._output_sizes[1:]
        self._nonlinearity = nonlinearity

        begin = 0
        for idx in range(self._num_layers-1):
            in_size = output_sizes[idx]
            out_size = output_sizes[idx+1]
            end = begin + in_size * out_size
            self.w_begin.append(begin)
            self.w_end.append(end)
            self.b_end.append(end+out_size)
            begin = end + out_size

    def __call__(self, inputs, weights):
        preds = inputs
        for idx in range(self._num_layers-2):
            w_shape = tf.concat([tf.shape(weights)[:-1], tf.constant([self.in_size[idx], self.out_size[idx]],dtype=tf.int32)],axis=0)
            b_shape = tf.concat([tf.shape(weights)[:-1], tf.constant([self.out_size[idx]],dtype=tf.int32)],axis=0)
            w = tf.reshape(
                weights[..., self.w_begin[idx]:self.w_end[idx]], w_shape)
            b = tf.reshape(
                weights[...,self.b_begin[idx]:self.b_end[idx]], b_shape)
            preds = tf.linalg.matvec(w, preds, transpose_a=True) + b
            preds = self._nonlinearity(preds)

        w_shape = tf.concat([tf.shape(weights)[:-1], tf.constant([self.in_size[idx+1], self.out_size[idx+1]],dtype=tf.int32)],axis=0)
        b_shape = tf.concat([tf.shape(weights)[:-1], tf.constant([self.out_size[idx+1]],dtype=tf.int32)],axis=0)
        w = tf.reshape(
            weights[..., self.w_begin[idx+1]:self.w_end[idx+1]], w_shape)
        b = tf.reshape(
            weights[...,self.b_begin[idx+1]:self.b_end[idx+1]], b_shape)
        preds = tf.linalg.matvec(w, preds, transpose_a=True) + b
        return preds

