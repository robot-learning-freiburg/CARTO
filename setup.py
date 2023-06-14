"""Install script for setuptools."""

import setuptools
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# TODO
# Add requirements.txt parsing

setuptools.setup(
    name="carto",
    version="0.0.1",
    author="Nick Heppert",
    author_email="heppert@cs.uni-freiburg.de",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)
