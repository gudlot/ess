# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# # flake8: noqa: E501
"""
Tests for tools module
"""

# author: Andrew R. McCluskey (arm61)

import unittest
from numpy.testing import assert_almost_equal
from ess.amor import tools


class TestQGrid(unittest.TestCase):
    def test_q_grid_a(self):
        actual = tools.q_grid(q_fix=0.05)
        expected = [
            0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275,
            0.03, 0.0325, 0.035, 0.0375, 0.04, 0.0425, 0.045, 0.0475, 0.05,
            0.0525, 0.055125, 0.05788125, 0.06077531, 0.06381408, 0.06700478,
            0.07035502, 0.07387277, 0.07756641, 0.08144473
        ]
        assert_almost_equal(actual, expected)

    def test_q_grid_b(self):
        actual = tools.q_grid(q_min=0.07, q_fix=0.1)
        expected = [0.07, 0.075, 0.08, 0.085]
        assert_almost_equal(actual, expected)

    def test_q_grid_c(self):
        actual = tools.q_grid(q_min=0.07, q_fix=0.001)
        expected = [0.06641707, 0.06973792, 0.07322482, 0.07688606, 0.08073037]
        assert_almost_equal(actual, expected)

    def test_q_grid_d(self):
        actual = tools.q_grid(d_q=0.0005)
        expected = [
            0.0073205, 0.00805255, 0.00885781, 0.00974359, 0.01071794,
            0.01178974, 0.01296871, 0.01426558, 0.01569214, 0.01726136,
            0.01898749, 0.02088624, 0.02297486, 0.02527235, 0.02779959,
            0.03057955, 0.0336375, 0.03700125, 0.04070137, 0.04477151,
            0.04924866, 0.05417353, 0.05959088, 0.06554997, 0.07210497,
            0.07931546
        ]
        assert_almost_equal(actual, expected)

    def test_q_grid_e(self):
        with self.assertRaises(ValueError):
            _ = tools.q_grid(q_min=0.7, q_max=0.001)
