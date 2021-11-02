# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
"""
Tests for resolution module
"""

# author: Andrew R. McCluskey (arm61)

import unittest
import numpy as np
import scipp as sc
from ess.reflectometry import resolution


class TestResolution(unittest.TestCase):
    def test_detector_resolution(self):
        expected_result = (np.arctan(1.0) * 180.0 / np.pi /
                           (2 * np.sqrt(2 * np.log(2)))) * sc.units.deg
        detector_spatial_resolution = 1 * sc.units.m
        z_pixel_position = 2 * sc.units.m
        z_sample_position = 1 * sc.units.m
        actual_result = resolution.detector_resolution(detector_spatial_resolution,
                                                       z_pixel_position,
                                                       z_sample_position)
        np.testing.assert_almost_equal(actual_result.values, expected_result.values)

    def test_z_offset(self):
        x = 0.0 * sc.units.m
        y = 0.0 * sc.units.m
        initial_z = 2.0 * sc.units.m
        expected_z = 0.0 * sc.units.m
        expected_result = sc.geometry.position(x, y, expected_z)
        actual_result = resolution.z_offset(sc.geometry.position(x, y, initial_z),
                                            -2 * sc.units.m)
        np.testing.assert_almost_equal(actual_result.values, expected_result.values)
