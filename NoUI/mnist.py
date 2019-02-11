import numpy as np
from keras.models import Model
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Reshape, Dropout, Activation
from keras.layers import Input, BatchNormalization
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.optimizers import Adam

from keras_capsnet.layer.capsnet import PrimaryCaps, Caps, ClassesCaps, Mask
from keras_capsnet.losses import margin
from keras_capsnet.activations import squash
from datasets.mnist import dataGenerator


num_class = 10
input_shape = (28, 28, 1)
m_train = 60000
m_test = 50000

batch_size = 32
epochs = 30

data_augmentation = {
    'width_shift_range': 2,
    'height_shift_range': 2
}


# Dataset

trainGenerator = dataGenerator('train', batch_size=batch_size, **data_augmentation)
testGenerator = dataGenerator('test', batch_size=batch_size)


# Model

x = Input(shape=input_shape)
y = Input(shape=(num_class,))

encoder = Convolution2D(filters=80, kernel_size=(9, 9), activation='relu') (x)
encoder = PrimaryCaps(capsules=8, capsule_dim=6, kernel_size=(9, 9), strides=2, activation='relu', activation_caps=squash) (encoder)
encoder = Caps(capsules=10, capsule_dim=10, routings=3, activation_caps=squash) (encoder)

output = ClassesCaps(name='capsule') (encoder)

decoder = Mask() (encoder, y_true=y)
decoder = Dense(512, activation='relu') (decoder)
decoder = Dense(1024, activation='relu') (decoder)
decoder = Dense(784, activation='sigmoid') (decoder)
decoder = Reshape((input_shape), name='reconstruction') (decoder)

model_training = Model(inputs=[x, y], outputs=[output, decoder])
model = Model(inputs=x, outputs=output)

model.summary()


# Training

model_training.compile(optimizer=Adam(lr=2e-3),
                       loss=[margin(), mean_squared_error],
                       loss_weights=[1, 0.0005],
                       metrics={'capsule': 'accuracy'})

model.compile(optimizer=Adam(), loss=margin(), metrics=['accuracy'])

hist = model_training.fit_generator(trainGenerator,
                                    steps_per_epoch=m_train // batch_size,
                                    epochs=epochs,
                                    validation_data=testGenerator,
                                    validation_steps=m_test // batch_size,
                                    verbose=1)

model.save_weights('model_tmp.h5')