# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import os
import shutil
import glob
import sys


class FileMover():
    def __init__(self, source_root, destination_root):
        self.source_root = source_root
        self.destination_root = destination_root

    def move_file(self, src, dst):
        os.write(1, "move {} {}\n".format(src, dst).encode())
        shutil.move(src, dst)

    def move(self, src, dst):
        src = os.path.join(self.source_root, *src)
        dst = os.path.join(self.destination_root, *dst)
        if '*' in dst:
            dst = glob.glob(dst)[-1]
        if '*' in src:
            for f in glob.glob(src):
                self.move_file(f, dst)
        else:
            self.move_file(src, dst)


if __name__ == '__main__':

    source_root = os.getcwd()
    destination_root = os.environ.get('CONDA_PREFIX')

    # Create a file mover to place the source files in the correct directories
    # for conda build.
    m = FileMover(source_root=source_root, destination_root=destination_root)

    # Depending on the platform, directories have different names.
    if sys.platform == "win32":
        lib_dest = 'lib'
    else:
        lib_dest = os.path.join('lib', 'python*')

    # Write fixed version to file to avoid having gitpython as a hard
    # dependency
    sys.path.append(os.path.abspath('src'))
    from ess._version import __version__ as v
    with open(os.path.join('src', 'ess', '_fixed_version.py'), 'w') as f:
        f.write(f'__version__ = \'{v}\'')

    m.move(['src', 'ess'], [lib_dest])
