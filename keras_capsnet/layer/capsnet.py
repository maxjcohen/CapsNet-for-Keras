# Coding: UTF-8
"""Capsule Network Layers.
"""

import numpy as np

from keras.engine.base_layer import Layer
from keras.layers.convolutional import Convolution2D
from keras import backend as K

class PrimaryCaps(Convolution2D):
    """PrimaryCaps layer as described in Hinton's paper"""
    def __init__(self, capsules, capsule_dim, **kwargs):
        super().__init__(filters=capsules*capsule_dim, **kwargs)
        self.capsule_dim = capsule_dim
        self.capsules = capsules

    def call(self, inputs):
        # Apply convolution
        outputs = super().call(inputs)
        outputs_shape = outputs.shape

        # Reshape -> (None, -1, capsule_dim)
        outputs = K.reshape(outputs, (K.shape(outputs)[0], outputs_shape[1]*outputs_shape[2]*self.capsules, self.capsule_dim))

        # Squash
        s_norm = K.sum(K.square(outputs), -1, keepdims=True)
        scale = s_norm / (1 + s_norm) / K.sqrt(s_norm + K.epsilon())
        return outputs * scale

    def compute_output_shape(self, input_shape):
        outputs_shape = super().compute_output_shape(input_shape)
        return (input_shape[0], outputs_shape[1]*outputs_shape[2]*self.capsules, self.capsule_dim)

class Caps(Layer):
    """Regular capsule layer. Input must be a PrimaryCaps layer. For example, see CapsDigit in original paper."""
    def __init__(self, capsules, capsule_dim, routings, kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.capsules = capsules
        self.capsule_dim = capsule_dim
        self.routings = routings
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor (PrimaryCaps) should have shape=[None, num_capsules, dim_capsule]"
        self.input_capsule_dim = input_shape[-1]
        self.input_capsules = input_shape[-2]

        self.W = self.add_weight(shape=(self.capsules, self.input_capsules, self.capsule_dim, self.input_capsule_dim),
                                  initializer=self.kernel_initializer,
                                  name='W',
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)

        self.built = True


    def call(self, inputs):
        # Brodcast inputs to match capsules number
        # inputs.shape:     batch_size, input_capsules, input_capsule_dim
        # u.shape           batch_size, capsules, input_capsules, input_capsule_dim
        u = K.expand_dims(inputs, 1)
        u = K.tile(u, [1, self.capsules, 1, 1])

        # Matrix multiplication along the last axis of u
        # u_hat.shape:      batch_size, capsules, input_capsules, capsule_dim
        u_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=u)

        # Routing algorithm
        # b.input_shape     batch_size, capsules, input_capsules
        b = K.zeros(shape=(K.shape(u_hat)[0], self.capsules, self.input_capsules))

        for r in range(self.routings):
            c = K.softmax(b, axis=1)

            # Weighted sum, and squash activation
            # s.shape       batch_size, capsules, dim_capsule
            s = K.batch_dot(c, u_hat, [2, 2])
            v = self.squash(s)

            if r > 0:
                b += K.batch_dot(v, u_hat, axes=[2, 3])

        return v

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.capsules, self.capsule_dim)

    def squash(self, s):
        s_norm = K.sum(K.square(s), -1, keepdims=True)
        scale = s_norm / (1 + s_norm) / K.sqrt(s_norm + K.epsilon())
        return s * scale

class Length(Layer):
    """Output of a capsule network, compute the norm of each capsule"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return K.sqrt(K.sum(K.square(inputs), axis=-1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

class Mask(Layer):
    """Mask all but correct capsule. Use for training with a decoder."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, y_true):
        mask = K.expand_dims(y_true, -1)

        return K.batch_flatten(inputs * mask)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * input_shape[2])
