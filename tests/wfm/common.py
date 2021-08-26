# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc


# TODO replace with sc.allclose after 0.8 scipp release
def allclose(x, y):
    return sc.all(sc.isclose(x, y)).value
