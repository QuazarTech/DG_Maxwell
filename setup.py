import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "DG_Maxwell",
    version = "0.0.0",
    author = "Aman Abhishek Tiwari, Balavarun P, Dr. Manichandra Morampudi",
    author_email = "amanabt@quazartech.com, mani@quazartech.com",
    description = ("An demonstration of how to create, document, and publish "
                                   "to the cheese shop a5 pypi.org."),
    license = "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    keywords = "physics discontinuous galerkin maxwell wave equation advection code",
    url = "https://github.com/QuazarTech/DG_Maxwell",
    packages=['dg_maxwell', 'tests'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    setup_requires = ['pytest-runner'],
    tests_require  = ['pytest']
)