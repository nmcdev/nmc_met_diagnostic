# _*_ coding: utf-8 _*_

from os import path
from setuptools import find_packages, setup
from codecs import open


name = "nmc_met_diagnostic"
author = __import__(name).__author__
version = __import__(name).__version__

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=name,

    version=version,

    description=("A collection of meteorological"
                 "diagnostic and analysis functions."),
    long_description=long_description,

    # author
    author=author,
    author_email='kan.dai@foxmail.com',

    # LICENSE
    license='GPL3',

    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],

    packages=find_packages(exclude=['docs', 'tests', 'build', 'dist']),
    include_package_data=True,
    exclude_package_data={'': ['.gitignore', '*.pyc', '*.pyo']},

    install_requires=['numpy>=1.12.1',
                      'scipy>=0.19.0',
                      'nmc_met_base'],
    dependency_links=[
       'git+https://github.com/nmcdev/nmc_met_base.git@master#egg=nmc_met_base',
    ]
)

# development mode (DOS command):
#     python setup.py develop
#     python setup.py develop --uninstall

# build mode：
#     python setup.py build --build-base=D:/test/python/build

# distribution mode:
#     python setup.py sdist             # create source tar.gz file in /dist
#     python setup.py bdist_wheel       # create wheel binary in /dist
