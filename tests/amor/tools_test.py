# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Andrew R. McCluskey (arm61)
import pytest
import scipp as sc
from ess.amor import tools


def test_q_grid_a():
    actual = tools.q_grid(q_fix=0.05 * sc.Unit('1/angstrom'))
    expected = sc.array(dims=['qz'],
                        values=[
                            0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025,
                            0.0275, 0.03, 0.0325, 0.035, 0.0375, 0.04, 0.0425, 0.045,
                            0.0475, 0.05, 0.0525, 0.055125, 0.05788125, 0.06077531,
                            0.06381408, 0.06700478, 0.07035502, 0.07387277, 0.07756641,
                            0.08144473
                        ],
                        unit='1/angstrom')
    assert sc.allclose(actual, expected)


def test_q_grid_b():
    actual = tools.q_grid(q_min=0.07 * sc.Unit('1/angstrom'),
                          q_fix=0.1 * sc.Unit('1/angstrom'))
    expected = sc.array(dims=['qz'],
                        values=[0.07, 0.075, 0.08, 0.085],
                        unit='1/angstrom')
    assert sc.allclose(actual, expected)


def test_q_grid_c():
    actual = tools.q_grid(q_min=0.07 * sc.Unit('1/angstrom'),
                          q_fix=0.001 * sc.Unit('1/angstrom'))
    expected = sc.array(
        dims=['qz'],
        values=[0.06641707, 0.06973792, 0.07322482, 0.07688606, 0.08073037],
        unit='1/angstrom')
    assert sc.allclose(actual, expected)


def test_q_grid_d():
    actual = tools.q_grid(d_q=0.0005 * sc.Unit('1/angstrom'))
    expected = sc.array(dims=['qz'],
                        values=[
                            0.0073205, 0.00805255, 0.00885781, 0.00974359, 0.01071794,
                            0.01178974, 0.01296871, 0.01426558, 0.01569214, 0.01726136,
                            0.01898749, 0.02088624, 0.02297486, 0.02527235, 0.02779959,
                            0.03057955, 0.0336375, 0.03700125, 0.04070137, 0.04477151,
                            0.04924866, 0.05417353, 0.05959088, 0.06554997, 0.07210497,
                            0.07931546
                        ],
                        unit='1/angstrom')
    assert sc.allclose(actual, expected)


def test_q_grid_e():
    with pytest.raises(ValueError):
        _ = tools.q_grid(q_min=0.7 * sc.Unit('1/angstrom'),
                         q_max=0.001 * sc.Unit('1/angstrom'))


def test_q_grid_unit():
    actual = tools.q_grid(q_min=0.7 * sc.Unit('1/nm'), q_fix=1. * sc.Unit('1/nm'))
    expected = sc.array(dims=['qz'],
                        values=[0.07, 0.075, 0.08, 0.085],
                        unit='1/angstrom')
    assert sc.allclose(actual, expected)
