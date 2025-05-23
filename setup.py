# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com), Swli (lucasleesw9@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function, division, absolute_import
"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages, Extension
# To use a consistent encoding
from codecs import open
from os import path
NO_PYBIND11 = False
try:
    import pybind11
except ModuleNotFoundError:
    NO_PYBIND11 = True

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

install_requires = [
    'torch>=2.0.0',
    'torchvision>=0.11.1',
    'transformers>4.20.0,<=4.47.0',
    'python-dateutil>=2.1',
    'psutil',
    'tensorboardX>=1.8',
    'datasets>=2.0.0',
    'huggingface-hub>=0.14.1',
]

# 定义扩展模块
if not NO_PYBIND11:
    ext_modules = [
        Extension(
            'autopipe',  # 模块名
            sources=['Merak/cpp/autopipe.cpp'],  # C++源文件列表
            include_dirs=[pybind11.get_include()],  # pybind11头文件路径
            language='c++',  # 指定C++语言
            extra_compile_args=['-O3'],  # 添加额外的编译选项，例如优化级别
        )
    ]

# https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package/16084844#16084844
# exec(open('Merak/version.py').read())
setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # There are some restrictions on what makes a valid project name
    name='Merak',  # Required

    version="2.2.2",  # Required

    # This is a one-line description or tagline of what your project does. 
    description='A framework for 3D parallelism',  # Required

    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    long_description=long_description,  # Optional

    # This should be a valid link to your project's main homepage.
    url='https://github.com/HPDL-Group/Merak',  # Optional

    # This should be your name or the name of the organization which owns the
    # project.
    author='HPDL group',  # Optional

    # Optional
    keywords='Large model training 3D parallelism pytorch GPT2 BERT',

    packages=find_packages(),  # Required

    install_requires=install_requires,

    # This should be a valid email address corresponding to the author listed
    # above.
    # author_email='',  # Optional

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        # 'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],

    ext_modules=None if NO_PYBIND11 else ext_modules,
)
