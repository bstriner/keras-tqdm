from tqdm import tqdm_notebook
from tqdm_callback import TQDMCallback
import sys


class TQDMNotebookCallback(TQDMCallback):
    def __init__(self, outer_description="Training",
                 inner_description_initial="Epoch {epoch}",
                 inner_description_update="[{metrics}] ",
                 metric_format="{name}: {value:0.3f}",
                 separator=", ",
                 file=sys.stderr):
        super(TQDMNotebookCallback, self).__init__(outer_description, inner_description_initial,
                                                   inner_description_update, metric_format, separator, file)

    def tqdm(self, desc, total):
        return tqdm_notebook(desc=desc, total=total)
