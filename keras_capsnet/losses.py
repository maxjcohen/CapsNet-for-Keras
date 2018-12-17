import keras.backend as K

def margin(margin=0.9, lambdas=0.5):
    def loss(y_true, y_pred):
        L_c = y_true * K.square(K.maximum(0.0, margin - y_pred)) + lambdas * (1 - y_true) * K.square(K.maximum(0.0, y_pred - (1.0 - margin)))
        return K.mean(K.sum(L_c, axis=1))
    return loss
