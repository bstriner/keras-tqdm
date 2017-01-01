from mnist_model import mnist_test
from keras_tqdm import TQDMCallback

if __name__ == "__main__":
    mnist_test(verbose=0, callbacks=[TQDMCallback()])
