from distutils.core import setup, Extension

# the c extension module
extension_mod = Extension("utils", sources = ["wrapper.c", "utils.c"])

setup(name = "UtilsModule", version = "0.1-dev",
      description = "Utils package around C",
      ext_modules=[extension_mod])