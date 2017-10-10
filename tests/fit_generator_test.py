import keras
import numpy as np
import pytest
from keras.layers import Dense
from keras.models import Sequential
from keras_tqdm import TQDMCallback


def fit_generator_test():
    m = Sequential()
    m.add(Dense(5, input_shape=(10,)))
    n = 1024
    _x = np.random.random((n, 10))
    _y = np.random.random((n, 5))
    epochs = 10
    m.compile("adam", "mean_squared_error")

    def generator():
        while True:
            idx = np.random.random_integers(0, _x.shape[0] - 1, (32,))
            yield _x[idx, ...], _y[idx, ...]

    if int(keras.__version__.split(".")[0]) == 2:
        m.fit_generator(generator(), steps_per_epoch=32 * 10, epochs=epochs, verbose=0, callbacks=[TQDMCallback()])
    else:
        m.fit_generator(generator(), nb_epoch=epochs, samples_per_epoch=32 * 10, verbose=0, callbacks=[TQDMCallback()])


if __name__ == "__main__":
    pytest.main([__file__])
