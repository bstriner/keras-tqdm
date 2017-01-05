from setuptools import setup, find_packages

setup(name='keras_tqdm',
      version='1.0.0',
      install_requires=['keras', 'tqdm'],
      author="Ben Striner",
      author_email="bstriner@gmail.com",
      url="https://github.com/bstriner/keras-tqdm",
      download_url='https://github.com/bstriner/keras-tqdm/tarball/v1.0.0',
      description="Train Keras models with TQDM progress bars and IPython support",
      keywords = ['keras', 'tqdm', 'progress', 'progressbar', 'ipython', 'jupyter'],
      packages=find_packages())
