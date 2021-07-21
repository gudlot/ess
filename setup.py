# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import os
import setuptools
import sys


def find_packages():
    # Write fixed version to file to avoid having gitpython as a hard
    # dependency
    sys.path.append(os.path.abspath('src'))
    from ess._version import __version__ as v
    with open(os.path.join('src', 'ess', '_fixed_version.py'), 'w') as f:
        f.write(f'__version__ = \'{v}\'\n')
    return setuptools.find_packages('src')


setuptools.setup(name='ess', packages=find_packages(), package_dir={"": "src"})
