"""
This example trains a model on the MNIST data set using keras-tqdm progress bars and fit_generator.
"""
from keras_tqdm import TQDMCallback
from mnist_model import mnist_model_generator

if __name__ == "__main__":
    mnist_model_generator(verbose=0, callbacks=[TQDMCallback()])
