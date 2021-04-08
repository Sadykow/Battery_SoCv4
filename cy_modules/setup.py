from distutils.core import setup
from Cython.Build import cythonize

setup(name = "UtilsModule", version = "0.2-dev",
      description = "Utils package around Cython",
      ext_modules=cythonize('*.pyx',
                        compiler_directives={'language_level' : "3"}
                  )
      )