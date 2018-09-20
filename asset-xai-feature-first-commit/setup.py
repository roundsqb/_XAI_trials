# QUANTUMBLACK CONFIDENTIAL
#
# Copyright (c) 2016 - present QuantumBlack Visual Analytics Ltd. All
# Rights Reserved.
#
# NOTICE: All information contained herein is, and remains the property of
# QuantumBlack Visual Analytics Ltd. and its suppliers, if any. The
# intellectual and technical concepts contained herein are proprietary to
# QuantumBlack Visual Analytics Ltd. and its suppliers and may be covered
# by UK and Foreign Patents, patents in process, and are protected by trade
# secret or copyright law. Dissemination of this information or
# reproduction of this material is strictly forbidden unless prior written
# permission is obtained from QuantumBlack Visual Analytics Ltd.

from setuptools import setup, find_packages
from os import path
import re

name = 'xai'
here = path.abspath(path.dirname(__file__))

# get package version
with open(path.join(here, name, '__init__.py'), encoding='utf-8') as f:
    version = re.search("__version__ = '([^']+)'", f.read()).group(1)

# get the dependencies and installs
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requires = [x.strip() for x in f if x.strip()]

# get test dependencies and installs
with open('test_requirements.txt', 'r', encoding='utf-8') as f:
    test_requires = [x.strip() for x in f
                     if x.strip() and not x.startswith('-r')]

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

setup(
    name=name,
    version=version,
    description='A library to generate post-hoc explanations.',
    long_description=readme,
    url='https://github.com/quantumblack/asset-xai',
    author='QuantumBlack',
    author_email='feedback@quantumblack.com',
    packages=find_packages(exclude=['docs*', 'tests*', 'tools*', 'scripts*']),
    tests_require=test_requires,
    install_requires=requires,
)
