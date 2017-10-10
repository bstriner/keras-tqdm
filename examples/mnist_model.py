import keras
import numpy as np
from keras.datasets import mnist
from keras.layers import Dropout, BatchNormalization, LeakyReLU, Dense, Input, Activation
from keras.models import Model
from keras.utils.np_utils import to_categorical


def build_model():
    x = Input((28 * 28,), name="x")
    hidden_dim = 512
    h = x
    h = Dense(hidden_dim)(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(0.2)(h)
    h = Dropout(0.5)(h)
    h = Dense(hidden_dim / 2)(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(0.2)(h)
    h = Dropout(0.5)(h)
    h = Dense(10)(h)
    h = Activation('softmax')(h)
    m = Model(x, h)
    m.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    return m


def mnist_process(x, y):
    return x.astype(np.float32).reshape((x.shape[0], -1)) / 255.0, to_categorical(y, 10)


def mnist_data():
    data = mnist.load_data()
    return [mnist_process(*d) for d in data]


def mnist_model(verbose=1, callbacks=[]):
    m = build_model()
    (xtrain, ytrain), (xtest, ytest) = mnist_data()
    if int(keras.__version__.split(".")[0]) == 2:
        m.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=10, batch_size=32, verbose=verbose,
              callbacks=callbacks)
    else:
        m.fit(xtrain, ytrain, validation_data=(xtest, ytest), nb_epoch=10, batch_size=32, verbose=verbose,
              callbacks=callbacks)


def mnist_generator(n):
    (xtrain, ytrain), (xtest, ytest) = mnist_data()
    while True:
        idx = np.random.random_integers(0, xtrain.shape[0] - 1, (n,))
        yield xtrain[idx, ...], ytrain[idx, ...]


def mnist_model_generator(verbose=1, callbacks=[]):
    m = build_model()
    (xtrain, ytrain), (xtest, ytest) = mnist_data()

    if int(keras.__version__.split(".")[0]) == 2:
        m.fit_generator(mnist_generator(32), validation_data=(xtest, ytest), epochs=10, steps_per_epoch=32 * 20,
                        verbose=verbose,
                        callbacks=callbacks)
    else:
        m.fit_generator(mnist_generator(32), validation_data=(xtest, ytest), nb_epoch=10, samples_per_epoch=32 * 20,
                        verbose=verbose,
                        callbacks=callbacks)


if __name__ == "__main__":
    mnist_model()
