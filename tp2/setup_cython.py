# compile with:
# python setup_cython.py build_ext --inplace


from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
  Extension(
        "im2col_cython",
        ["im2col_cython.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules = cythonize(extensions),
)