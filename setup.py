#!/usr/bin/env python

# Building:
# python setup.py build_ext
#
# Installing package from sources:
# python setup.py install
# For developers (creating a link to the sources):
# python setup.py develop
#
# Building source distribution:
# python setup.py sdist
# The generated source file is needed for installing by a tool like pip (pip install ...).


# from distutils.core import setup
from setuptools import setup
from setuptools import find_packages
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np
import os


copt = {'msvc': ['-IpyNumpyCythonExample/cpp/external', '/EHsc']}  # set additional include path and EHsc exception handling for VS
lopt = {}


class build_ext_opt(build_ext):
    def initialize_options(self):
        build_ext.initialize_options(self)
        self.compiler = 'msvc' if os.name == 'nt' else None  # in Anaconda the libpython package includes the MinGW import libraries and a file (Lib/distutils/distutils.cfg) which sets the default compiler to mingw32. Alternatively try conda remove libpython.

    def build_extensions(self):
        c = self.compiler.compiler_type
        if c in copt:
            for e in self.extensions:
                e.extra_compile_args = copt[c]
        if c in lopt:
            for e in self.extensions:
                e.extra_link_args = lopt[c]
        build_ext.build_extensions(self)


extensions = [
    Extension('pyNumpyCythonExample.ArrayInterface', ['pyNumpyCythonExample/cpp/ArrayInterface.pyx'])
]


f = open('VERSION', 'r')
version = f.readline().strip()
f.close()

author = 'David-Leon Pohl'
author_email = 'pohl@physik.uni-bonn.de'

setup(
    name='pyTestbeamAnalysis',
    version=version,
    description='A simple test beam analysis in Python.',
    url='https://github.com/SiLab-Bonn/pyTestbeamAnalysis',
    license='BSD 3-Clause ("BSD New" or "BSD Simplified") License',
    long_description='',
    author=author,
    maintainer=author,
    author_email=author_email,
    maintainer_email=author_email,
    install_requires=['cython', 'numpy'],
    packages=find_packages(),  # exclude=['*.tests', '*.test']),
    include_package_data=True,  # accept all data files and directories matched by MANIFEST.in or found in source control
    package_data={'': ['*.txt', 'VERSION'], 'docs': ['*'], 'examples': ['*']},
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
    cmdclass={'build_ext': build_ext_opt},
    platforms='any'
)

try:
    import pyNumpyCythonExample.ArrayInterface as ai
    hist = ai.get_hist()
    print hist
    print hist.flags
    print "STATUS: SUCCESS!"
except Exception, e:
    print "STATUS: FAILED (%s)" % str(e)