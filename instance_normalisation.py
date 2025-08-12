# =========================================================================
# === FINAL VERSION of InstanceNormalization to match the old .h5 file ===
# =========================================================================
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
import tensorflow as tf

class InstanceNormalization(Layer):
    """
    A special version of Instance Normalization modified to load weights
    from an old model that incorrectly saved gamma/beta with shape (1,).
    """
    def __init__(self,
                 axis=-1,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        # =========================================================
        # === THIS IS THE CRUCIAL MODIFICATION ===
        # We are forcing the shape of gamma and beta to be (1,) to
        # match the shape of the weights saved in the .h5 file.
        shape = (1,)
        # =========================================================

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
            
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
            
        self.built = True

    def call(self, inputs, training=None):
        input_shape = tf.keras.backend.int_shape(inputs)
        reduction_axes = [i for i in range(len(input_shape)) if i not in (0, self.axis)]
        mean = tf.keras.backend.mean(inputs, reduction_axes, keepdims=True)
        variance = tf.keras.backend.var(inputs, reduction_axes, keepdims=True)
        normalized = (inputs - mean) / tf.keras.backend.sqrt(variance + self.epsilon)

        if self.scale:
            normalized = self.gamma * normalized
        if self.center:
            normalized = normalized + self.beta
            
        return normalized

    def get_config(self):
        # ... (get_config can remain the same) ...
        config = {
            'axis': self.axis, 'epsilon': self.epsilon, 'center': self.center,
            'scale': self.scale, 'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))