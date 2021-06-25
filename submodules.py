import tensorflow as tf
import sonnet as snt

# class forward_initialiser(snt.Module):
#     def __init__(self, initial_state_type, dim_reprs, num_classes, nn_size, nn_layers, float_dtype, dropout_rate, initialiser, nonlinearity, name="forward_initialiser"):
#         super(forward_initialiser, self).__init__(name=name)
#         self._initial_state_type = initial_state_type

#         if self._initial_state_type == tf.constant("zero", dtype=tf.string):
#             self.initialiser = constant_initialiser(
#                 dim_reprs=dim_reprs,
#                 float_dtype=float_dtype,
#                 num_classes=num_classes,
#                 trainable=False)
#         elif self._initial_state_type == tf.constant("constant", dtype=tf.string):
#             self.initialiser = constant_initialiser(
#                 dim_reprs=dim_reprs,
#                 float_dtype=float_dtype,
#                 num_classes=num_classes,
#                 trainable=True)
#         elif self._initial_state_type == tf.constant("parametric", dtype=tf.string):
#             self.initialiser = parametric_initialiser(
#                 nn_size=nn_size,
#                 nn_layers=nn_layers,
#                 dim_reprs=dim_reprs,
#                 num_classes=num_classes,
#                 dropout_rate=dropout_rate,
#                 initialiser=initialiser,
#                 nonlinearity=nonlinearity,
#             )
#         else:
#             raise NameError("Unknown initial state type")

#     def __call__(self, x, is_training=True):
#         num_points = tf.shape(x)[-2]
#         print(num_points)
#         if self._initial_state_type == tf.constant("parametric",dtype=tf.string):
#             reprs = self.initialiser(x, is_training=is_training)
#         else:
#             #reprs = self.initialiser(num_points)
#             reprs = self.initialiser(x, is_training=is_training)
#         return reprs



class constant_initialiser(snt.Module):
    def __init__(self, dim_reprs, float_dtype, num_classes, no_batch=False, trainable=False, name="constant_initialiser"):
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

        self._num_classes = num_classes

        if no_batch:
            self.tile_fun = lambda init, x : tf.tile(init, [x.shape[-2], self._num_classes])
        else:
            def tile_fun_batch(init, x):
                t = tf.tile(init, [x.shape[-2], self._num_classes])
                return tf.stack([t for i in range(x.shape[-3])])

            self.tile_fun = tile_fun_batch
        
    def __call__(self, x):
        return self.tile_fun(self.init, x)
        #tf.tile(self.init, [num_points, self._num_classes])


class parametric_initialiser(snt.Module):
    def __init__(self, nn_size, nn_layers, dim_reprs, num_classes, dropout_rate, initialiser, nonlinearity, name="parametric_initialiser"):
        super(parametric_initialiser, self).__init__(name=name)
        self._num_classes = num_classes
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


class neural_local_updater(snt.Module):
    def __init__(self, nn_size, nn_layers, dim_reprs, num_classes, initialiser, nonlinearity, no_batch=False, name="neural_local_updater"):
        super(neural_local_updater, self).__init__(name=name)
        self._num_classes = num_classes

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

    def __call__(self, r, y, x=None, iter=""):
        y = tf.one_hot(y, self._num_classes) #TODO: move to data preprocessing
        y = tf.transpose(y, self.perm)
        outputs = self.module1(r)
        agg_outputs = tf.math.reduce_mean(outputs, axis=-2, keepdims=True)
        outputs = tf.concat([outputs, tf.tile(agg_outputs, self.tile_dim)], axis=-1)

        outputs_t = self.module2(outputs)

        outputs_f = self.module3(outputs)
        outputs = outputs_t * y + outputs_f * (1-y)
        return outputs


class decoder(snt.Module):
    def __init__(self, embedding_dim, orthogonality_penalty_weight, initialiser, name="decoder"):
        super(decoder, self).__init__(name=name)
        self.orthogonality_reg = get_orthogonality_regularizer(orthogonality_penalty_weight)
        self.decoder_module = snt.Linear(
            output_size=embedding_dim * 2,
            with_bias=True,
            w_init=initialiser,
            name="decoder"
        )

    def __call__(self, inputs):
        outputs = self.decoder_module(inputs)
        return outputs, self.orthogonality_reg(self.decoder_module.w)
        




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
            name="orthogonality_regularisation"
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
        self.initialiser = tf.keras.initializers.GlorotUniform()

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

    def __call__(self, x1, x2, r):
        if self._rep == tf.constant("identity", dtype=tf.string):
            k, q = (x1, x2)
        else: # mapping a
            k = self.module(x1)
            q = self.module(x2)

        return dot_product_attention(q, k, r, self._normalise)

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
        weights = tf.math.softmax(unnorm_weights)
    else:
        weights = tf.math.sigmoid(unnorm_weights)
    rep = tf.linalg.matmul(weights, v)
    return rep

class deep_se_kernel(snt.Module): #TODO: clarify whether nn_layer or embedding dim should be used for nerual layer width
    def __init__(self, embedding_layers, kernel_dim, initialiser, nonlinearity, name="deep_se_kernel"):
        super(deep_se_kernel, self).__init__(name=name)
        self.module = snt.nets.MLP( # mapping a
            output_sizes=[kernel_dim] * embedding_layers,
            w_init=initialiser,
            with_bias=True,
            activation=nonlinearity,
            name="deep_se_kernel"
        )

    def __call__(self, querys, keys, values, sigma, lengthscale):
        keys = self.module(keys)
        querys = self.module(querys)
        return squared_exponential_kernel(querys, keys, values, sigma, lengthscale)


# def squared_exponential_kernel(querys, keys, values, sigma, lengthscale):
#     """rbf kernel"""
#     print("query",querys.shape)
#     print("key", keys.shape)
#     print("value",values.shape)
#     num_keys = tf.shape(keys)[-2]
#     num_querys = tf.shape(querys)[-2]

#     _keys = tf.tile(tf.expand_dims(keys, axis=1), [1, num_querys, 1])
#     _querys = tf.tile(tf.expand_dims(querys, axis=0), [num_keys, 1, 1])
#     sq_norm = tf.reduce_sum((_keys - _querys)**2, axis=-1)
#     kernel_qk = sigma**2 * tf.math.exp(- sq_norm / (2.*lengthscale**2)) # RBF
#     v = tf.linalg.matmul(kernel_qk, values, transpose_a=True) # RBF * V
#     return v

def squared_exponential_kernel(querys, keys, values, sigma, lengthscale):
    """rbf kernel"""
    sq_norm = tf.reduce_sum((tf.expand_dims(keys, -3) - tf.expand_dims(querys, -2))**2, axis=-1)
    sq_norm = tf.linalg.matrix_transpose(sq_norm)
    kernel_qk = sigma**2 * tf.math.exp(- sq_norm / (2.*lengthscale**2)) # RBF
    v = tf.linalg.matmul(kernel_qk, values, transpose_a=True) # RBF * V
    return v








    


