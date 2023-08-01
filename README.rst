PyColorimetry
=============

PyColorimetry is a powerful Python library designed for both educators and students in the field of colorimetry. The library processes images using semantic segmentation. After segmentation, the images are normalized, and computations of RGB, tristimulus XYZ values, and conversion to the CIELAB space are performed. PyColorimetry also provides functionality for visualizing colors in the CIELAB color space. This library takes advantage of modern GPU computing power to provide efficient and accurate colorimetric computations. PyColorimetry aims to make complex colorimetric concepts more accessible, enabling deeper understanding and fostering innovation in color science.

|Python| |Pandas| |Numpy| |Matplotlib| |Scipy| |Skimage| |Sklearn| |Colab| |Torch|

.. |Python| image:: https://img.shields.io/badge/python%20-%2314354C.svg?&style=flat&logo=python&logoColor=white
  :target: https://www.python.org/
  :alt: Python

.. |Pandas| image:: https://img.shields.io/badge/Pandas%20-2C2D72?style=flat&logo=pandas&logoColor=white
  :target: https://pandas.pydata.org/
  :alt: Pandas

.. |Numpy| image:: https://img.shields.io/badge/numpy%20-%230095D5.svg?&style=flat&logo=numpy&logoColor=white
  :target: https://numpy.org/
  :alt: Numpy

.. |Matplotlib| image:: https://img.shields.io/badge/Matplotlib%20-008080?style=flat&logo=matplotlib&logoColor=white
  :target: https://matplotlib.org/
  :alt: Matplotlib

.. |Scipy| image:: https://img.shields.io/badge/scipy%20-00599C?style=flat&logo=scipy&logoColor=white
  :target: https://scipy.org/
  :alt: Scipy

.. |Skimage| image:: https://img.shields.io/badge/skimage%20--FFAD00?style=flat&logo=scikit-image&logoColor=white
  :target: https://scikit-image.org/
  :alt: Skimage

.. |Sklearn| image:: https://img.shields.io/badge/Sklearn%20-F7931E?style=flat&logo=scikit-learn&logoColor=white
  :target: https://scikit-learn.org/
  :alt: Sklearn

.. |Colab| image:: https://img.shields.io/badge/Colab%20--FFAD00?style=flat&logo=googlecolab&logoColor=white
  :target: https://colab.research.google.com/
  :alt: Colab

.. |Torch| image:: https://img.shields.io/badge/Torch%20-EE4C2C?style=flat&logo=pytorch&logoColor=white
  :target: https://pytorch.org/
  :alt: Torch

Installation 
============

The PyColorimetry library may be installed using pip:
  
.. code:: python

    !git clone https://github.com/josorio398/PyColorimetry_Library /content/PyColorimetry_Library
    !pip install -r /content/PyColorimetry_Library/requirements.txt

You also need to download the weights for the SAM model:

.. code:: python

    !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

To import the library, you can use:

.. code:: python

    from PyColorimetry.ColorimetricAnalysis import *

Requirements
============

- Python 3.6 or later
- GPU support
- Libraries: Pandas, Numpy, Matplotlib, Scipy, Skimage, Sklearn, Torch
- Models: SAM
- Installation support is currently provided for Google Colab

Maintainer
==========

- **Prof. Jhonny Osorio Gallego, M.Sc.**

https://github.com/josorio398

Universidad de América

jhonny.osorio@profesores.uamerica.edu.co


