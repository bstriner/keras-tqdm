from keras.callbacks import Callback
from tqdm import tqdm
import sys
from math import ceil


class TQDMCallback(Callback):
    def __init__(self, description="Training", epoch_description="Epoch {}", file=sys.stderr):
        self.description = description
        self.epoch_description = epoch_description
        self.file = file
        self.tqdm_outer = None
        self.tqdm_inner = None

    def on_epoch_begin(self, epoch, logs={}):
        batch_count = int(ceil(self.params['nb_sample'] / self.params['batch_size']))
        self.tqdm_inner = tqdm(desc=self.epoch_description.format(epoch), total=batch_count)

    def on_epoch_end(self, epoch, logs={}):
        self.tqdm_inner.close()
        self.tqdm_outer.update(1)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.tqdm_inner.update(1)

    def on_train_begin(self, logs={}):
        self.tqdm_outer = tqdm(desc=self.description, total=self.params["nb_epoch"])

    def on_train_end(self, logs={}):
        self.tqdm_outer.close()
