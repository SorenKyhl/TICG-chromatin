import subprocess

from pybind11.setup_helpers import Pybind11Extension
from setuptools import Extension, find_packages, setup

module1 = Pybind11Extension(name = 'pylib.pyticg',
                    include_dirs = ['include'],
                    language = 'c++',
                    sources = ["src/pybind_Sim.cpp"],
                    extra_compile_args=["-g"])

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
