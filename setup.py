from setuptools import find_packages, setup, Extension
import subprocess
from pybind11.setup_helpers import Pybind11Extension

module1 = Pybind11Extension(name = 'pylib.pyticg',
                    include_dirs = ['include'],
                    language = 'c++',
                    sources = ["src/pybind_Sim.cpp"])

setup(
    name='pylib',
    packages=find_packages(include=['pylib', 'pylib.*']),
    version='0.1.3',
    description='set up library',
    author='Soren Kyhl',
    license='MIT',
	install_requires=['hic-straw','jsbeautifier'],#, 'hicrep', 'numba'],
    ext_modules=[module1],
)
