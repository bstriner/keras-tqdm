# keras-tqdm
Keras integration with TQDM progress bars.
* TQDM supports nested progress bars. If you have Keras fit and predict loops within an outer TQDM loop, the
nested loops will display properly.
* TQDM supports Jupyter/IPython notebooks.

##Installation
``` 
git clone https://github.com/bstriner/keras-tqdm.git
cd keras-tqdm
python setup.py install
```

##Keras
[Keras](https://github.com/fchollet/keras) is an awesome machine learning library for Theano
 or TensorFlow.

##TQDM
[TQDM](https://github.com/tqdm/tqdm) is a progress bar library with good support for nested loops and Jupyter/IPython notebooks.

##Basic TQDM Usage

Use TQDM to wrap enumerators within loops to create a progress bar. Review TQDM documentation for
display options.

```
from tqdm import tqdm
import time
for i in tqdm(range(10)):
    time.sleep(1)
```

##Keras TQDM

Use `keras_tqdm` to utilize TQDM progress bars for Keras fit loops. Keras `fit` loops nested inside TQDM `for` loops 
will display correctly.

```
from keras_tqdm import TQDMCallback
from tqdm import tqdm
for i in tqdm(range(10)):
    model.fit(x, y, callbacks = [TQDMCallback()])
```
