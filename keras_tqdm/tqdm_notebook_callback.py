import sys

from tqdm import tqdm_notebook

from .tqdm_callback import TQDMCallback


class TQDMNotebookCallback(TQDMCallback):
    def __init__(self,
                 outer_description="Training",
                 inner_description_initial="Epoch {epoch}",
                 inner_description_update="[{metrics}] ",
                 metric_format="{name}: {value:0.3f}",
                 separator=", ",
                 leave_inner=False,
                 leave_outer=True,
                 output_file=sys.stderr,
                 initial=0, **kwargs):
        super(TQDMNotebookCallback, self).__init__(outer_description=outer_description,
                                                   inner_description_initial=inner_description_initial,
                                                   inner_description_update=inner_description_update,
                                                   metric_format=metric_format,
                                                   separator=separator,
                                                   leave_inner=leave_inner,
                                                   leave_outer=leave_outer,
                                                   output_file=output_file,
                                                   initial=initial, **kwargs)

    def tqdm(self, desc, total, leave, initial=0):
        """
        Extension point. Override to provide custom options to tqdm_notebook initializer.
        :param desc: Description string
        :param total: Total number of updates
        :param leave: Leave progress bar when done
        :return: new progress bar
        :param initial: Initial counter state
        """
        return tqdm_notebook(desc=desc, total=total, leave=leave, initial=initial)
