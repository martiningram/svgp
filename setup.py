from os import getenv
from setuptools import setup
from setuptools import find_packages


setup(
    name='svgp',
    version=getenv("VERSION", "LOCAL"),
    description='Experiments with stochastic VI for GPs',
    packages=find_packages()
)
