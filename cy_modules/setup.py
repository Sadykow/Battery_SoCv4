from distutils.core import setup
from Cython.Build import cythonize

setup(name = "UtilsModule", version = "0.1-dev",
      description = "Utils package around Cython",
      ext_modules=cythonize('utils.pyx'))