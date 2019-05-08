import keras.backend as K

def squash(x, axis=-1):
    s_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_norm) / (0.5 + s_norm)
    return scale * x