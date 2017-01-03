from keras.callbacks import Callback
from tqdm import tqdm
import sys
from math import ceil
import six
import numpy as np


class TQDMCallback(Callback):
    def __init__(self, outer_description="Training",
                 inner_description_initial="Epoch: {epoch}",
                 inner_description_update="Epoch: {epoch} - {metrics}",
                 metric_format="{name}: {value:0.3f}",
                 separator=", ",
                 file=sys.stderr):
        """
        Construct a callback that will create and update progress bars.
        :param outer_description: string for outer progress bar
        :param inner_description_initial: initial format for epoch ("Epoch: {epoch}")
        :param inner_description_update: format after metrics collected ("Epoch: {epoch} - {metrics}")
        :param metric_format: format for each metric name/value pair ("{name}: {value:0.3f}")
        :param separator: separator between metrics (", ")
        :param file:
        """
        self.outer_description = outer_description
        self.inner_description_initial = inner_description_initial
        self.inner_description_update = inner_description_update
        self.metric_format = metric_format
        self.separator = separator
        self.file = file
        self.tqdm_outer = None
        self.tqdm_inner = None
        self.epoch = None
        self.running_logs = None

    def tqdm(self, desc, total):
        return tqdm(desc=desc, total=total)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
        desc = self.inner_description_initial.format(epoch=self.epoch)
        batch_count = int(ceil(self.params['nb_sample'] / self.params['batch_size']))
        self.tqdm_inner = self.tqdm(desc=desc, total=batch_count + 1)
        self.running_logs = {}

    def on_epoch_end(self, epoch, logs={}):
        metrics = self.format_metrics(logs)
        desc = self.inner_description_update.format(epoch=epoch, metrics=metrics)
        self.tqdm_inner.desc = desc
        # set miniters and mininterval to 0 so last update displays
        self.tqdm_inner.miniters = 0
        self.tqdm_inner.mininterval = 0
        self.tqdm_inner.update(1)
        self.tqdm_inner.close()
        self.tqdm_outer.update(1)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.append_logs(logs)
        metrics = self.format_metrics(self.running_logs)
        desc = self.inner_description_update.format(epoch=self.epoch, metrics=metrics)
        self.tqdm_inner.desc = desc
        self.tqdm_inner.update(1)

    def on_train_begin(self, logs={}):
        self.tqdm_outer = self.tqdm(desc=self.outer_description, total=self.params["nb_epoch"])

    def on_train_end(self, logs={}):
        self.tqdm_outer.close()

    def append_logs(self, logs):
        metrics = self.params['metrics']
        for metric, value in six.iteritems(logs):
            if metric in metrics:
                if metric in self.running_logs:
                    self.running_logs[metric].append(value[()])
                else:
                    self.running_logs[metric] = [value[()]]

    def format_metrics(self, logs):
        metrics = self.params['metrics']
        strings = [self.metric_format.format(name=metric, value=np.mean(logs[metric], axis=None)) for metric in metrics
                   if
                   metric in logs]
        return self.separator.join(strings)
