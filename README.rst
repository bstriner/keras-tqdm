keras-tqdm
==========

Keras integration with TQDM progress bars.

* `Keras <https://github.com/fchollet/keras>`__ is an awesome machine learning library for Theano or TensorFlow.
* `TQDM <https://github.com/tqdm/tqdm>`__ is a progress bar library with good support for nested loops and Jupyter/IPython notebooks.

Key features
------------

* TQDM supports nested progress bars. If you have Keras fit and
  predict loops within an outer TQDM loop, the nested loops will
  display properly.

* TQDM supports Jupyter/IPython notebooks.

* TQDM looks great!

``TQDMNotebookCallback`` with ``leave_inner=False`` (default)



.. figure:: https://github.com/bstriner/keras-tqdm/raw/master/docs/images/leave_inner_False.png
   :alt: Keras TQDM leave_inner=False

``TQDMNotebookCallback`` with ``leave_inner=True``

.. figure:: https://github.com/bstriner/keras-tqdm/raw/master/docs/images/leave_inner_True.png
   :alt: Keras TQDM leave_inner=True

``TQDMCallback`` for command-line scripts

.. figure:: https://github.com/bstriner/keras-tqdm/raw/master/docs/images/console.png
   :alt: Keras TQDM CLI
   
Installation
------------

Stable release
::

    pip install keras-tqdm


Development release
::

   pip install git+https://github.com/bstriner/keras-tqdm.git --upgrade --no-deps

Development mode (changes to source take effect without reinstalling)
::

    git clone https://github.com/bstriner/keras-tqdm.git
    cd keras-tqdm
    python setup.py develop

Basic usage
-----------

It's very easy to use Keras TQDM. The only required change is to remove default messages (`verbose=0`) and add a callback to ``model.fit``. The rest happens automatically! For Jupyter Notebook required code modification is as simple as:

::

    from keras_tqdm import TQDMNotebookCallback
    # keras, model definition...
    model.fit(X_train, Y_train, verbose=0, callbacks=[TQDMNotebookCallback()])

For plain text mode (e.g. for Python run from command line)

::

    from keras_tqdm import TQDMCallback
    # keras, model definition...
    model.fit(X_train, Y_train, verbose=0, callbacks=[TQDMCallback()])


Advanced usage
--------------

Use ``keras_tqdm`` to utilize TQDM progress bars for Keras fit loops.
``keras_tqdm`` loops can be nested inside TQDM loops to display nested progress bars (although you can use them
inside ordinary for loops as well).
Set ``verbose=0`` to suppress the default progress bar.

::

    from keras_tqdm import TQDMCallback
    from tqdm import tqdm
    for model in tqdm(models, desc="Training several models"):
        model.fit(x, y, verbose=0, callbacks=[TQDMCallback()])

For IPython and Jupyter notebook ``TQDMNotebookCallback`` instead of ``TQDMCallback``. Use ``tqdm_notebook`` in your own code instead of ``tqdm``.
Formatting is controlled by python format strings. The default ``metric_format`` is ``"{name}: {value:0.3f}"``.
For example, use  ``TQDMCallback(metric_format="{name}: {value:0.6f}")`` for 6 decimal points or ``{name}: {value:e}`` for scientific notation.

Questions?
----------

Please feel free to submit PRs and issues. Comments, questions, and
requests are welcome. If you need more control, subclass
``TQDMCallback`` and override the ``tqdm`` function.
