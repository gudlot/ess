# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)


class Beamline:
    def __init__(self, choppers, source):
        self.choppers = choppers
        self.source = source
