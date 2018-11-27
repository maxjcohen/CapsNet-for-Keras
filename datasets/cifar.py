from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from .utils import reshapeBatch

def dataGenerator(flag='train', batch_size=32, reshape=True, **kwargs):
    num_class = 10
    input_shape = (32, 32, 3)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(-1, *input_shape).astype('float32') / 255
    x_test = x_test.reshape(-1, *input_shape).astype('float32') / 255

    y_train = to_categorical(y_train, num_class)
    y_test = to_categorical(y_test, num_class)

    datagen = ImageDataGenerator(**kwargs)

    if flag == 'train':
        datagen.fit(x_train)
        generator = datagen.flow(x_train, y_train, batch_size)
    elif flag == 'test':
        generator = datagen.flow(x_test, y_test, batch_size)
    else:
        raise NameError(f'Unknown flag "{flag}" encountered in dataGenerator')

    if reshape:
        generator = reshapeBatch(generator)

    return generator
