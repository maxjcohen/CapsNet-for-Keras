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

        # Reshape -> (None, -1, capsule_dim)
        outputs = K.reshape(outputs, (K.shape(outputs)[0], -1, self.capsule_dim))

        # Squash
        s_norm = K.sum(K.square(outputs), -1, keepdims=True)
        scale = s_norm / (1 + s_norm) / K.sqrt(s_norm + K.epsilon())
        return outputs * scale

    def compute_output_shape(self, input_shape):
        outputs_shape = super().compute_output_shape(input_shape)
        return (input_shape[0], outputs_shape[1]*outputs_shape[2]*self.capsules, self.capsule_dim)
