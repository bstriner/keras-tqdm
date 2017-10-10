"""
This example trains a model on the MNIST data set using keras-tqdm progress bars.
"""
from keras_tqdm import TQDMCallback
from mnist_model import mnist_model

if __name__ == "__main__":
    mnist_model(verbose=0, callbacks=[TQDMCallback()])
