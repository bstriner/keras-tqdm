from sys import stderr

import numpy as np
import six
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm


class TQDMCallback(Callback):
    def __init__(self, outer_description="Training",
                 inner_description_initial="Epoch: {epoch}",
                 inner_description_update="Epoch: {epoch} - {metrics}",
                 metric_format="{name}: {value:0.3f}",
                 separator=", ",
                 leave_inner=True,
                 leave_outer=True,
                 show_inner=True,
                 show_outer=True,
                 output_file=stderr,
                 initial=0):
        """
        Construct a callback that will create and update progress bars.

        :param outer_description: string for outer progress bar
        :param inner_description_initial: initial format for epoch ("Epoch: {epoch}")
        :param inner_description_update: format after metrics collected ("Epoch: {epoch} - {metrics}")
        :param metric_format: format for each metric name/value pair ("{name}: {value:0.3f}")
        :param separator: separator between metrics (", ")
        :param leave_inner: True to leave inner bars
        :param leave_outer: True to leave outer bars
        :param show_inner: False to hide inner bars
        :param show_outer: False to hide outer bar
        :param output_file: output file (default sys.stderr)
        :param initial: Initial counter state
        """
        self.outer_description = outer_description
        self.inner_description_initial = inner_description_initial
        self.inner_description_update = inner_description_update
        self.metric_format = metric_format
        self.separator = separator
        self.leave_inner = leave_inner
        self.leave_outer = leave_outer
        self.show_inner = show_inner
        self.show_outer = show_outer
        self.output_file = output_file
        self.tqdm_outer = None
        self.tqdm_inner = None
        self.epoch = None
        self.running_logs = None
        self.inner_count = None
        self.initial = initial

    def tqdm(self, desc, total, leave, initial=0):
        """
        Extension point. Override to provide custom options to tqdm initializer.
        :param desc: Description string
        :param total: Total number of updates
        :param leave: Leave progress bar when done
        :param initial: Initial counter state
        :return: new progress bar
        """
        return tqdm(desc=desc, total=total, leave=leave, file=self.output_file, initial=initial)

    def build_tqdm_outer(self, desc, total):
        """
        Extension point. Override to provide custom options to outer progress bars (Epoch loop)
        :param desc: Description
        :param total: Number of epochs
        :return: new progress bar
        """
        return self.tqdm(desc=desc, total=total, leave=self.leave_outer, initial=self.initial)

    def build_tqdm_inner(self, desc, total):
        """
        Extension point. Override to provide custom options to inner progress bars (Batch loop)
        :param desc: Description
        :param total: Number of batches
        :return: new progress bar
        """
        return self.tqdm(desc=desc, total=total, leave=self.leave_inner)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
        desc = self.inner_description_initial.format(epoch=self.epoch)
        self.mode = 0  # samples
        if 'samples' in self.params:
            self.inner_total = self.params['samples']
        elif 'nb_sample' in self.params:
            self.inner_total = self.params['nb_sample']
        else:
            self.mode = 1  # steps
            self.inner_total = self.params['steps']
        if self.show_inner:
            self.tqdm_inner = self.build_tqdm_inner(desc=desc, total=self.inner_total)
        self.inner_count = 0
        self.running_logs = {}

    def on_epoch_end(self, epoch, logs={}):
        metrics = self.format_metrics(logs)
        desc = self.inner_description_update.format(epoch=epoch, metrics=metrics)
        if self.show_inner:
            self.tqdm_inner.desc = desc
            # set miniters and mininterval to 0 so last update displays
            self.tqdm_inner.miniters = 0
            self.tqdm_inner.mininterval = 0
            self.tqdm_inner.update(self.inner_total - self.tqdm_inner.n)
            self.tqdm_inner.close()
        if self.show_outer:
            self.tqdm_outer.update(1)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if self.mode == 0:
            update = logs['size']
        else:
            update = 1
        self.inner_count += update
        if self.inner_count < self.inner_total:
            self.append_logs(logs)
            metrics = self.format_metrics(self.running_logs)
            desc = self.inner_description_update.format(epoch=self.epoch, metrics=metrics)
            if self.show_inner:
                self.tqdm_inner.desc = desc
                self.tqdm_inner.update(update)

    def on_train_begin(self, logs={}):
        if self.show_outer:
            epochs = (self.params['epochs'] if 'epochs' in self.params
                      else self.params['nb_epoch'])
            self.tqdm_outer = self.build_tqdm_outer(desc=self.outer_description,
                                                    total=epochs)

    def on_train_end(self, logs={}):
        if self.show_outer:
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
