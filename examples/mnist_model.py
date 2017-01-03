from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dropout, BatchNormalization, LeakyReLU, Dense, Input, Activation
import numpy as np
from keras.utils.np_utils import to_categorical


def build_model():
    x = Input((28 * 28,), name="x")
    hidden_dim = 512
    h = x
    h = Dense(hidden_dim)(h)
    h = BatchNormalization(mode=0)(h)
    h = LeakyReLU(0.2)(h)
    h = Dropout(0.5)(h)
    h = Dense(hidden_dim / 2)(h)
    h = BatchNormalization(mode=0)(h)
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
    m.fit(xtrain, ytrain, validation_data=(xtest, ytest), nb_epoch=10, batch_size=32, verbose=verbose,
          callbacks=callbacks)


if __name__ == "__main__":
    mnist_model()
