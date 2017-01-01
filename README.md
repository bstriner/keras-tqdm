# keras-tqdm
Keras integration with TQDM progress bars. One particular advantage is that TQDM supports
nested progress bars. If you have Keras fit and predict loops within an outer TQDM loop, the
nested loops will display properly.

##Keras
[Keras](https://github.com/fchollet/keras) is an awesome machine learning library for Theano
 or TensorFlow. Does not provide extensive customization of progress bars.



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

