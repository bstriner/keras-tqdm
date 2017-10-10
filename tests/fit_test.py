import keras
import numpy as np
import pytest
from keras.layers import Dense
from keras.models import Sequential
from keras_tqdm import TQDMCallback


def fit_test():
    m = Sequential()
    m.add(Dense(5, input_shape=(10,)))
    n = 1024
    _x = np.random.random((n, 10))
    _y = np.random.random((n, 5))
    epochs = 10
    m.compile("adam", "mean_squared_error")
    if int(keras.__version__.split(".")[0]) == 2:
        m.fit(_x, _y, epochs=epochs, verbose=0, callbacks=[TQDMCallback()])
    else:
        m.fit(_x, _y, nb_epoch=epochs, verbose=0, callbacks=[TQDMCallback()])


if __name__ == "__main__":
    pytest.main([__file__])
