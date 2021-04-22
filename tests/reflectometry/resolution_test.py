"""
Tests for resolution module
"""

# author: Andrew R. McCluskey (arm61)

import unittest
from numpy.testing import assert_almost_equal
import scipp as sc
from ess.reflectometry import resolution


class TestResolution(unittest.TestCase):
    def test_detector_resolution(self):
        expected_result = 19.1097408 * sc.units.deg
        detector_spatial_resolution = 1 * sc.units.m
        z_pixel_position = 2 * sc.units.m
        z_sample_position = 1 * sc.units.m
        actual_result = resolution.detector_resolution(
            detector_spatial_resolution, z_pixel_position, z_sample_position)
        assert_almost_equal(actual_result.values, expected_result.values)

    def test_z_offset(self):
        x = 0.0 * sc.units.m
        y = 0.0 * sc.units.m
        initial_z = 2.0 * sc.units.m
        expected_z = 0.0 * sc.units.m
        expected_result = sc.geometry.position(x, y, expected_z)
        actual_result = resolution.z_offset(
            sc.geometry.position(x, y, initial_z), -2 * sc.units.m)
        assert_almost_equal(actual_result.values, expected_result.values)
