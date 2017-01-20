from setuptools import setup, find_packages
import re

long_description = re.sub("\r\n", "\n", open('README.rst').read())
setup(name='keras_tqdm',
      version='1.0.6',
      install_requires=['Keras', 'tqdm'],
      author="Ben Striner",
      author_email="bstriner@gmail.com",
      url="https://github.com/bstriner/keras-tqdm",
      download_url='https://github.com/bstriner/keras-tqdm/tarball/v1.0.6',
      description="Keras models with TQDM progress bars in Jupyter notebooks",
      long_description=long_description,
      keywords=['keras', 'tqdm', 'progress', 'progressbar', 'ipython', 'jupyter'],
      license='MIT',
      classifiers=[
          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3'
      ],
      packages=find_packages())
