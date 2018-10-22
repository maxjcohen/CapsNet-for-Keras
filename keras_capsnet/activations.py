import keras.backend as K

def squash(x):
    s_norm = K.sum(K.square(x), -1, keepdims=True)
    scale = s_norm / (1 + s_norm) / K.sqrt(s_norm + K.epsilon())
    return x * scale