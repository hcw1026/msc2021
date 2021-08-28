import math
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import time


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
    def __init__(self, nn_size, nn_layers, dim_reprs, initialiser, nonlinearity, num_iters=1, indp_iter=False, classification=True, num_classes=1, no_batch=False, xNone=False, name="neural_local_updater"):
        super(neural_local_updater, self).__init__(name=name)

        self._num_iters = num_iters if indp_iter else 1

        if classification:
            self._num_classes = num_classes
            self._dim_reprs = dim_reprs

            # MLP m
            self.module1_list = [
                snt.nets.MLP(
                output_sizes=[nn_size] * nn_layers,
                w_init=initialiser,
                with_bias=True,
                activation=nonlinearity,
                name="neural_local_updater_m_{}".format(i)) for i in range(self._num_iters)]

            # MLP u+
            self.module2_list = [
                snt.nets.MLP(
                output_sizes=[nn_size] * nn_layers + [dim_reprs],
                w_init=initialiser,
                with_bias=True,
                activation=nonlinearity,
                name="neural_local_updater_uplus_{}".format(i)) for i in range(self._num_iters)]

            # MLP u-
            self.module3_list = [
                snt.nets.MLP(
                output_sizes=[nn_size] * nn_layers + [dim_reprs],
                w_init=initialiser,
                with_bias=True,
                activation=nonlinearity,
                name="neural_local_updater_uminus_{}".format(i)) for i in range(self._num_iters)]

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

            self.module_list = [
                snt.nets.MLP(
                output_sizes=[nn_size] * nn_layers + [dim_reprs],
                w_init=initialiser,
                with_bias=True,
                activation=nonlinearity,
                name="neural_local_updater_{}".format(i)) for i in range(self._num_iters)]

            self.call_fn = self.regression        

    def classification(self, r, y, x=None, iteration=0):
        r_shape = tf.shape(r)
        r = tf.reshape(r, tf.concat([r_shape[:-1], tf.constant([self._num_classes, self._dim_reprs], dtype=tf.int32)], axis=0))

        y = tf.one_hot(y, self._num_classes) #TODO: move to data preprocessing
        y = tf.transpose(y, self.perm)
        outputs = self.module1_list[iteration](r)
        agg_outputs = tf.math.reduce_mean(outputs, axis=-2, keepdims=True)
        outputs = tf.concat([outputs, tf.tile(agg_outputs, self.tile_dim)], axis=-1)

        outputs_t = self.module2_list[iteration](outputs)

        outputs_f = self.module3_list[iteration](outputs)
        outputs = outputs_t * y + outputs_f * (1-y)
        outputs = tf.reshape(outputs, shape=r_shape)
        return outputs

    def regression(self, r, y, x, iteration=0):
        reprs = self.concat_fn(r, y, x)
        return self.module_list[iteration](reprs)

    def __call__(self, r, y, x=None, iteration=0):
        return self.call_fn(r=r, y=y, x=x, iteration=min(self._num_iters-1, iteration))


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
        elif repr_as_inputs is False:
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
  
    def __init__(self, config, num_iters=1, indp_iter=False, complete_return=True, name="attention"):
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

        self._num_iters = num_iters if indp_iter else 1

        if self._rep == "mlp":
            self.module_list = [
                snt.nets.MLP( # mapping a
                output_sizes=self._output_sizes,
                w_init=self.initialiser,
                with_bias=True,
                activation=self._nonlinearity,
                name="deep_attention_{}".format(i)
                ) for i in range(self._num_iters)]

        if self._att_type != tf.constant("dot_product",tf.string):
            raise NameError("Unknown attention type")

        if self._rep not in [tf.constant("identity", dtype=tf.string), tf.constant("mlp", dtype=tf.string)]:
            raise NameError("Unknown attention representation - not among ['identity', 'mlp']")

        if self._rep == tf.constant("identity", dtype=tf.string):
            self.call_fn_frontend = lambda keys, querys, iteration: dot_product_attention_frontend_fn(keys=keys, querys=querys, normalise=self._normalise)
        else:
            self.call_fn_frontend = lambda keys, querys, iteration: dot_product_attention_frontend_fn(keys=self.module_list[iteration](keys), querys=self.module_list[iteration](querys), normalise=self._normalise)
        
        if complete_return:
            self.call_fn_backend = lambda weights, values: dot_product_attention_backend_fn(weights=weights, values=values)
        else:
            self.call_fn_backend = lambda weights, values: weights


    def __call__(self, keys, querys, recompute, precomputed, values=None, iteration=0):
        if recompute:
            weights = self.call_fn_frontend(keys=keys, querys=querys, iteration=min(self._num_iters-1, iteration))
        else:
            weights = precomputed
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

    def __call__(self, querys, keys, sigma, lengthscale, recompute, precomputed, values=None, iteration=0):
        if recompute:
            query_key = self.call_fn_frontend(querys=querys, keys=keys, sigma=sigma, lengthscale=lengthscale)
        else:
            query_key = precomputed
        return self.call_fn_backend(query_key=query_key, values=values)

    def backend(self, query_key, values):
        return squared_exponential_kernel_backend_fn(query_key=query_key, values=values)

#### Deep squared-exponential kernel
class deep_se_kernel(snt.Module): #TODO: clarify whether nn_layer or embedding dim should be used for nerual layer width
    def __init__(self, embedding_layers, kernel_dim, initialiser, nonlinearity, num_iters=1, indp_iter=False, complete_return=True, name="deep_se_kernel"):
        super(deep_se_kernel, self).__init__(name=name)

        self._num_iters = num_iters if indp_iter else 1

        self.module_list = [
            snt.nets.MLP( # mapping a
            output_sizes=[kernel_dim] * embedding_layers,
            w_init=initialiser,
            with_bias=True,
            activation=nonlinearity,
            name="deep_se_kernel_{}".format(i)
        ) for i in range(self._num_iters)]

        self.call_fn_frontend = lambda querys, keys, sigma, lengthscale, iteration: squared_exponential_kernel_frontend_fn(
            querys=self.module_list[iteration](querys), keys=self.module_list[iteration](keys), sigma=sigma, lengthscale=lengthscale)

        if complete_return:
            self.call_fn_backend = lambda query_key, values: squared_exponential_kernel_backend_fn(query_key=query_key, values=values)
        else:
            self.call_fn_backend = lambda query_key, values: query_key

    def __call__(self, querys, keys, sigma, lengthscale, recompute, precomputed, values=None, iteration=0):
        if recompute:
            query_key = self.call_fn_frontend(querys=querys, keys=keys, sigma=sigma, lengthscale=lengthscale, iteration=min(self._num_iters-1, iteration))
        else:
            query_key = precomputed
        return self.call_fn_backend(query_key=query_key, values=values)

    def backend(self, query_key, values):
        return squared_exponential_kernel_backend_fn(query_key=query_key, values=values)

#################################################################################
# Sampling
#################################################################################

def deterministic_sample(distribution_params, stddev_offset):
    """deterministic sampling by splitting parameteres into mean and stddev directly"""
    #means, unnormalised_stddev = tf.split(distribution_params, 2, axis=-1)
    #return means
    return distribution_params

def probabilistic_sample(distribution_params, stddev_offset, is_training=True):
    """probabilistic sampling"""
    means, unnormalized_stddev = tf.split(distribution_params, 2, axis=-1) # mean and log std
    stddev = tf.math.exp(unnormalized_stddev) - (1. - stddev_offset)
    stddev = tf.math.maximum(stddev, 1.e-10)
    stddev = tf.math.minimum(stddev, 1.e5)
    if not is_training:
        return means
    else:
        return tf.random.normal(shape=tf.shape(means), mean=means, stddev=stddev)


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

class predict_repr_as_inputs_simple(snt.Module):
    def __init__(self, nn_size, nn_layers, output_dim, initialiser, nonlinearity, name="predict"):
        super(predict_repr_as_inputs_simple, self).__init__(name=name)
        self.module = snt.nets.MLP(
                output_sizes=[nn_size] * nn_layers + [2 * output_dim],
                activation=nonlinearity,
                with_bias=True,
                w_init=initialiser,
                name="predict_simple"
            )

    def __call__(self, inputs, weights):
        return self.module(weights)

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


#################################################################################
# Fourier Features
#################################################################################

class FourierFeatures(snt.Module):
    def __init__(self, stddev_init, embedding_dim, num_freq, learnable, num_iters, indp_iter, float_dtype=tf.float32, name="fourier_features"):
        super(FourierFeatures, self).__init__(name=name)
        self._num_iters = num_iters if indp_iter else 1

        self.freq_mat = [tf.Variable(
            initial_value=tf.random.normal(shape=[embedding_dim, num_freq], mean=0, stddev=stddev_init, dtype=float_dtype), 
            trainable=learnable, 
            name=name+"_"+str(i)) for i in range(self._num_iters)]
        self.embedding_dim = 2 * num_freq

    def __call__(self, X, recompute, precomputed, iteration=0):
        if recompute:
            X_tr = tf.linalg.matmul(X, self.freq_mat[min(self._num_iters-1, iteration)])
            cos_tr = tf.math.cos(2 * math.pi * X_tr)
            sin_tr = tf.math.sin(2* math.pi * X_tr)
            return tf.concat([cos_tr, sin_tr], axis=-1)
        else:
            return precomputed



#################################################################################
# Deep sets
#################################################################################

class PermEqui(snt.Module): #not support batch
    def __init__(self, dim_out, initialiser, pool_fn="sum", name="perm_equi"):
        """
        dim_out: int
            - dimension of each output vector
        """
        super(PermEqui, self).__init__(name=name)

        self.Gamma = snt.Linear(output_size=dim_out)
        self.Lambda = snt.Linear(output_size=dim_out, with_bias=False, w_init=initialiser)

        if pool_fn == "sum":
            self.pool_fn = tf.math.reduce_max
        elif pool_fn == "mean":
            self.pool_fn = tf.math.reduce_mean
        else:
            raise NameError("pool_fn must be one of 'sum' and 'mean' ")

    def __call__(self, x):
        xm = self.pool_fn(x, axis=-2, keepdims=True)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm
        return x

class DeepSet(snt.Module):
    def __init__(self, dim_out, n_layers, pool_fn, dim_pre_tr, initialiser, name="deep_set"):
        super(DeepSet, self).__init__(name=name)

        dim_embed = dim_pre_tr if dim_pre_tr is not None else dim_out
        self.permequi = [
            PermEqui(dim_out=dim_embed, initialiser=initialiser, pool_fn=pool_fn, name="perm_equi_{}".format(i)) 
            for i in range(n_layers)]

        self._nonlinearity = tf.nn.elu
        self._n_layers = n_layers

        if dim_pre_tr is not None:
            self.pre_transform = snt.Linear(
                output_size=dim_pre_tr,
                with_bias=True,
                w_init=initialiser,
                name="linear_pre_tr"
                )

            self.pro_transform = snt.Linear(
                output_size=dim_out,
                with_bias=True,
                w_init=initialiser,
                name="linear_pro_tr"
                )

            self.frontend_fn = lambda x: self.pre_transform(x)
            self.backend_fn = lambda x: self.pro_transform(x)
        else:
            self.frontend_fn = lambda x: x
            self.backend_fn = lambda x: x

    def __call__(self, x):
        x = self.frontend_fn(x)
        for i in range(self._n_layers - 1):
            x = self.permequi[i](x)
            x = self._nonlinearity(x)
        
        x = self.permequi[-1](x)
        x = self.backend_fn(x)
        return x
        #x = tf.reduce_mean(x, axis=0)
        #x = tf.reduce_mean(self.permequi[-1](x),axis=-1)
        #return self.decoder(x)

def rff_kernel_frontend_fn(w, keys, querys, weights=1.):
    keys_tr = tf.linalg.matmul(tf.expand_dims(keys, -3), w, transpose_b=True) #(batch, 1, num_keys, num_w)
    querys_tr = tf.linalg.matmul(tf.expand_dims(querys, -2), w, transpose_b=True) #(batch, num_querys, 1, num_w)
    output = tf.concat([tf.math.sin(keys_tr) * weights, tf.math.cos(keys_tr) * weights], axis=-1) * tf.concat([tf.math.sin(querys_tr) * weights, tf.math.cos(querys_tr) * weights], axis=-1) #(batch, num_querys, num_keys, num_w)
    output = tf.reduce_mean(output, axis=-1) * 2. #(batch, num_querys, num_keys)
    return output
    #diff = tf.expand_dims(keys, -3) - tf.expand_dims(querys, -2)
    #prod = tf.linalg.matmul(diff, w, transpose_b=True)
    #tf.print(prod.shape)
    #return tf.reduce_mean(tf.math.square(tf.concat([tf.math.sin(prod), tf.math.cos(prod)], axis=-1)),axis=-1)

def rff_kernel_backend_fn(query_key, values):
    return tf.linalg.matmul(query_key, values) #(batch, num_querys, dim_values)

class rff_kernel(snt.Module):
    def __init__(self, dim_init, mapping, initialiser, dropout_rate=0., embedding_dim=None, transform_dim=None, float_dtype=tf.float32, init_distr="normal", init_distr_param=dict(), rff_init_trainable=True, rff_weight_trainable=False, num_iters=1, indp_iter=False, complete_return=True, name="rff_kernel"):
        """
        mapping: None or DeepSet etc.
        embedding_dim: int or None
            - original embedding dim of features. If None, features are transformed to transform_dim (transform_dim must be provided)
        transform_dim: int or None
            - the features linear transformation dimension if not None. If None, no transformation is performed
        """
        super(rff_kernel, self).__init__(name=name)

        self._num_iters = num_iters if indp_iter else 1

        if mapping is None:
            self.module_list = [lambda x: x] * self._num_iters
        else:
            self.module_list = [mapping(name="mapping_{}".format(i)) for i in range(self._num_iters)]

        if init_distr.lower() == "normal":
            param = {"mean":0., "stddev":1.}
            param.update(init_distr_param)
            random_initialiser = tf.random_normal_initializer(**param)
        elif init_distr.lower() == "uniform":
            param = {"minval":-1, "maxval":1}
            param.update(init_distr_param)
            random_initialiser = tf.random_uniform_initializer(**param)

        assert (transform_dim is not None) or (embedding_dim is not None), "at least one of transform_dim or embedding_dim must not be None"

        if transform_dim is not None:
            self.features_transform_list = [
                snt.Linear(
                    output_size=transform_dim,
                    with_bias=True,
                    w_init=initialiser,
                    name="linear_feature_transform_{}".format(i)
                ) for i in range(self._num_iters)
            ]
            embedding_dim = transform_dim
        else:
            self.features_transform_list = [lambda x: x] * self._num_iters

        self.rff_init_list = [
            tf.Variable(
                initial_value=random_initialiser(
                    shape=(dim_init, embedding_dim),
                    dtype=float_dtype),
                trainable=rff_init_trainable,
                name="rff_init_{}".format(i)
                ) for i in range(self._num_iters)]

        self.rff_weights = [
            tf.Variable(
                initial_value=tf.constant_initializer(value=1.)(
                shape=[dim_init],
                dtype=float_dtype), 
            trainable=rff_weight_trainable,
            name="rff_weights".format(i)
            ) for i in range(self._num_iters)]

        dropout = tf.keras.layers.GaussianDropout(rate=dropout_rate)

        self.call_fn_frontend = lambda querys, keys, iteration, is_training: rff_kernel_frontend_fn(
            w=dropout(self.module_list[iteration](self.rff_init_list[iteration]), training=is_training), querys=self.features_transform_list[iteration](querys), keys=self.features_transform_list[iteration](keys), weights=self.rff_weights[iteration])

        if complete_return:
            self.call_fn_backend = lambda query_key, values: rff_kernel_backend_fn(query_key=query_key, values=values)
        else:
            self.call_fn_backend = lambda query_key, values: query_key

    def __call__(self, querys, keys, recompute, precomputed, values=None, iteration=0, is_training=True):
        if recompute:
            query_key = self.call_fn_frontend(querys=querys, keys=keys, iteration=min(self._num_iters-1, iteration), is_training=is_training)
        else:
            query_key = precomputed
        return self.call_fn_backend(query_key=query_key, values=values)

    def backend(self, query_key, values):
        return rff_kernel_backend_fn(query_key=query_key, values=values)

class MultiHeadAttention(snt.Module):
    def __init__(self, dim_out, num_heads, initialiser, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self._num_heads = num_heads
        self._dim_out = dim_out
        self._dim_splits = self._dim_out // self._num_heads

        assert self._dim_out % self._num_heads == 0, "dim_out must be a multiple of num_heads"

        self.Wq = snt.Linear(
            output_size=dim_out,
            with_bias=True,
            w_init=initialiser,
            name="linear_Wq"
            )

        self.Wk = snt.Linear(
            output_size=dim_out,
            with_bias=True,
            w_init=initialiser,
            name="linear_Wk"
            )

        self.Wv = snt.Linear(
            output_size=dim_out,
            with_bias=True,
            w_init=initialiser,
            name="linear_Wv"
            )

        self.Wo = snt.Linear(
            output_size=dim_out,
            with_bias=True,
            w_init=initialiser,
            name="linear_Wo"
        )

    def __call__(self, querys, keys, values):

        querys = self.Wq(querys) #(num_querys, dim_out)
        keys = self.Wk(keys) #(num_keys, dim_out)
        values = self.Wv(values) #(num_keys, dim_out)

        querys = self.split(value=querys, size=self._num_heads) #(num_heads, num_querys, dim_out//num_heads)
        keys = self.split(value=keys, size=self._num_heads) #(num_heads, num_keys, dim_out//num_heads)
        values = self.split(value=values, size=self._num_heads) #(num_heads, num_keys, dim_out//num_heads)

        query_key = tf.linalg.matmul(querys, keys, transpose_b=True) / tf.math.sqrt(tf.cast(self._dim_out, dtype=tf.float32)) #(num_heads, num_querys, num_keys)
        query_key = tf.math.softmax(logits=query_key, axis=-1)
        output = tf.linalg.matmul(query_key, values) #(num_heads, num_querys, dim_out//num_heads)
        output =  tf.concat(tf.unstack(value=output, axis=0), axis=-1) #(num_querys, dim_out)
        return self.Wo(output)

    def split(self, value, size):
        return tf.stack(values=tf.split(value=value, num_or_size_splits=size, axis=-1), axis=0)

class MAB(snt.Module):
    def __init__(self, dim_out, dim_pre_tr, nn_layers, num_heads, initialiser, nonlinearity, float_dtype, name="MAB"):
        super(MAB, self).__init__(name=name)

        dim_embed = dim_pre_tr if dim_pre_tr is not None else dim_out
        self.multihead = MultiHeadAttention(dim_out=dim_embed, num_heads=num_heads, initialiser=initialiser)

        self.rFF = snt.nets.MLP(
            output_sizes=[dim_embed] * nn_layers, #set as the same size as dim_out
            w_init = initialiser,
            with_bias=True,
            activation=nonlinearity,
            activate_final=nonlinearity,
            name="MAB_rFF"
        )
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1.e-5, dtype=float_dtype, name="ln1")
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1.e-5, dtype=float_dtype, name="ln2")

        if dim_pre_tr is not None:
            self.pre_transform = snt.Linear(
                output_size=dim_pre_tr,
                with_bias=True,
                w_init=initialiser,
                name="linear_pre_tr"
                )

            self.pro_transform = snt.Linear(
                output_size=dim_out,
                with_bias=True,
                w_init=initialiser,
                name="linear_pro_tr"
                )

            self.frontend_fn = lambda x: self.pre_transform(x)
            self.backend_fn = lambda x: self.pro_transform(x)
        else:
            self.frontend_fn = lambda x: x
            self.backend_fn = lambda x: x


    def __call__(self, x, y):
        x = self.frontend_fn(x)
        H = self.ln1(x + self.multihead(querys=x, keys=y, values=y))
        output =  self.ln2(H + self.rFF(H))
        return self.backend_fn(output)

class SAB(snt.Module):
    def __init__(self, dim_out, dim_pre_tr, n_layers, nn_layers, num_heads, initialiser, nonlinearity, float_dtype, name="SAB"):
        """
        dim_pre_tr: int or None
             - If not None, pretransform to dimension dim_pre_tr (useful for multihead attention to make divisible number of heads)
        n_layers: int
            - number of stacked SAB
        nn_layers: int
            - number of feedforward layers inside a MAB rFF
        """
        super(SAB, self).__init__(name=name)
        self._n_layers = n_layers
        self.MAB_list = [
            MAB(
                dim_out=dim_out, dim_pre_tr=dim_pre_tr, nn_layers=nn_layers, num_heads=num_heads, initialiser=initialiser, nonlinearity=nonlinearity, float_dtype=float_dtype, name="MAB_{}".format(i)
            ) for i in range(n_layers)
            ]

    def __call__(self, x):
        for i in range(self._n_layers):
            x = self.MAB_list[i](x=x, y=x)
        return x

class ISAB(snt.Module):
    def __init__(self, dim_out, dim_pre_tr, n_layers, n_induce_points, nn_layers, num_heads, initialiser, nonlinearity, float_dtype, name="SAB"):
        super(ISAB, self).__init__(name=name)
        self._n_layers = n_layers
        self._n_induce_points = n_induce_points
        
        self.inducing_points = tf.Variable(
            tf.keras.initializers.GlorotNormal()(
                dtype=float_dtype,
                shape=[n_induce_points, dim_out]),
            trainable=True,
            name="inducing_points")

        self.MAB1_list = [
            MAB(
                dim_out=dim_out, dim_pre_tr=dim_pre_tr, nn_layers=nn_layers, num_heads=num_heads, initialiser=initialiser, nonlinearity=nonlinearity, float_dtype=float_dtype, name="MAB1_{}".format(i)
            ) for i in range(n_layers)
            ]

        self.MAB2_list = [
            MAB(
                dim_out=dim_out, dim_pre_tr=dim_pre_tr, nn_layers=nn_layers, num_heads=num_heads, initialiser=initialiser, nonlinearity=nonlinearity, float_dtype=float_dtype, name="MAB2_{}".format(i)
            ) for i in range(n_layers)
            ]

    def __call__(self, x):
        for i in range(self._n_layers):
            H = self.MAB1_list[i](x=self.inducing_points, y=x)
            x = self.MAB2_list[i](x=x, y=H)
        return x




#################################################################################
# Global latents
#################################################################################

class sample_latent(snt.Module):
    def __init__(self, num_z_samples, nn_layers, nn_size, dim_latent, stddev_const_scale, initialiser, nonlinearity, name="sample_latent"):
        super(sample_latent, self).__init__(name=name)

        self.latent_encoder = snt.nets.MLP(
            output_sizes=[nn_size] * nn_layers + [2 * dim_latent], #for mean and variance
            w_init=initialiser,
            with_bias=True,
            activation=nonlinearity,
            name="latent_encoder")

        self._stddev_const_scale = stddev_const_scale
        self._num_z_samples = num_z_samples

    def __call__(self, repr):
        repr_sum = tf.math.reduce_sum(repr, axis=-2, keepdims=True) #(batch_size, 1, dim_repr)
        repr_sum = self.latent_encoder(repr_sum) #(batch_size, 1, 2*dim_latent)
        mu, sigma = tf.split(repr_sum, 2, axis=-1) #(batch_size, 1, dim_latent)
        sigma = self._stddev_const_scale + (1-self._stddev_const_scale) * tf.nn.softplus(sigma)
        return tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

class latent_decoder(snt.Module):
    def __init__(self, initialiser, dim_reprs, nonlinearity, no_decoder, name="latent_decoder"):
        super(latent_decoder, self).__init__(name=name)
        self.merge_r_z = snt.Linear(
                output_size=dim_reprs,
                with_bias=True,
                w_init=initialiser,
                name="linear_merge_r_z"
                )

        if no_decoder:
            self._nonlinearity = lambda x:x
        else:
            self._nonlinearity = nonlinearity

    def __call__(self, R, z_samples):
        num_z_samples = tf.shape(z_samples)[0]
        R = tf.repeat(tf.expand_dims(R,axis=0), axis=0, repeats=num_z_samples) #(num_z_samples, batch_size, num_target, dim_reprs)
        z_samples = tf.repeat(z_samples, axis=-2, repeats=tf.shape(R)[-2]) #(num_z_samples, batch_size, num_target, dim_latent)
        output = tf.concat([R, z_samples], axis=-1) #(num_z_samples, batch_size, num_target, dim_latent+dim_reprs)
        return self._nonlinearity(self.merge_r_z(output))  #(num_z_samples, batch_size, num_target, dim_reprs)



