# Coding: UTF-8
"""Capsule Network Layers.
"""

import numpy as np

from keras.engine.base_layer import Layer
from keras.layers.convolutional import Convolution2D
from keras import backend as K
import tensorflow as tf

class CapsuleLayer(Layer):
    def __init__(self,
                 capsules,
                 capsule_dim,
                 activation_caps,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

        self.capsules = capsules
        self.capsule_dim = capsule_dim
        self.activation_caps = activation_caps
        
class PrimaryCaps(CapsuleLayer, Convolution2D):
    """PrimaryCaps layer as described in Hinton's paper"""
    def __init__(self, capsules, capsule_dim, activation_caps, **kwargs):
        super().__init__(capsules, capsule_dim, activation_caps, filters=capsules*capsule_dim, **kwargs)
    
    def call(self, inputs):
        # Apply convolution
        outputs = Convolution2D.call(self, inputs)
        outputs_shape = outputs.shape

        # Reshape -> (None, -1, capsule_dim)
        outputs = K.reshape(outputs, (K.shape(outputs)[0], outputs_shape[1]*outputs_shape[2]*self.capsules, self.capsule_dim))
        
        if self.activation_caps:
            return self.activation_caps(outputs)
        
        return outputs

    def compute_output_shape(self, input_shape):
        outputs_shape = Convolution2D.compute_output_shape(self, input_shape)
        return (input_shape[0], outputs_shape[1]*outputs_shape[2]*self.capsules, self.capsule_dim)

class Caps(CapsuleLayer):
    """Regular capsule layer. Input must be a PrimaryCaps layer. For example, see CapsDigit in original paper."""
    def __init__(self, capsules, capsule_dim, routings, activation_caps, **kwargs):
        super().__init__(capsules, capsule_dim, activation_caps, **kwargs)

        self.routings = routings
        

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
            v = self.activation_caps(s)

            b += K.batch_dot(v, u_hat, axes=[2, 3])

        return v

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.capsules, self.capsule_dim)
    
    
    
class CapsCNN(Convolution2D):
    def __init__(self, capsule_dim, routings, activation_caps, **kwargs):
        kwargs.setdefault("use_bias", False)
        super().__init__(filters=capsule_dim, **kwargs)
        
        self.capsule_dim = capsule_dim
        self.activation_caps = activation_caps
        self.routings = routings
        
    def call(self, inputs):
        KS = self.kernel_size[0]
        C = K.shape(inputs)[-1]
        _, SO, _, _ = self.compute_output_shape(inputs.shape)
        
        
        x_cols = tf.image.extract_image_patches(inputs,
                                       ksizes=[1, *self.kernel_size, 1],
                                       strides=[1, *self.strides, 1],
                                       rates=[1, *self.dilation_rate, 1],
                                       padding=self.padding.upper()
                                      )
        
        x_cols = K.reshape(x_cols, (-1, KS*KS, C))
        w = K.reshape(self.kernel, (KS*KS, C, self.capsule_dim))
        
        u_hat = K.local_conv1d(x_cols, w, [1], [1])
        u_hat = K.reshape(u_hat, (-1, SO, SO, KS*KS, self.capsule_dim))
        
        b = K.zeros(shape=(K.shape(u_hat)[0], SO, SO, KS*KS))
        
        for r in range(self.routings):
            c = K.softmax(b, axis=-1)
            
            v = self.activation_caps(K.batch_dot(c, u_hat))
            
            if r < self.routings - 1:
                b += K.batch_dot(v, u_hat, [3, 4])
        
        return v

class ClassesCaps(Layer):
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
