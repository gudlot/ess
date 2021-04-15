"""
Tests for corrections module
"""

# author: Andrew R. McCluskey (arm61)

import os
import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import scipp as sc
from ess.reflectometry import corrections

np.random.seed(1)
N = 9
VALUES = np.ones(N)
DETECTORS = np.random.randint(1, 5, size=(N))

DATA = sc.DataArray(
    data=sc.Variable(
        dims=["event"],
        unit=sc.units.counts,
        values=VALUES,
        dtype=sc.dtype.float32,
    ),
    coords={
        "detector_id": sc.Variable(
            dims=["event"], values=DETECTORS, dtype=sc.dtype.int32
        )
    },
)

DETECTOR_ID = sc.Variable(
    dims=["detector_id"], values=np.arange(1, 5), dtype=sc.dtype.int32
)
BINNED = sc.bin(DATA, groups=[DETECTOR_ID])

PIXELS = np.array([[1, 1, 1], [1, 2, 1], [2, 1, 1], [2, 2, 1]])
X = sc.Variable(
    dims=["detector_id"],
    values=PIXELS[:, 0],
    dtype=sc.dtype.float64,
    unit=sc.units.m,
)
Y = sc.Variable(
    dims=["detector_id"],
    values=PIXELS[:, 1],
    dtype=sc.dtype.float64,
    unit=sc.units.m,
)
Z = sc.Variable(
    dims=["detector_id"],
    values=PIXELS[:, 2],
    dtype=sc.dtype.float64,
    unit=sc.units.m,
)
BINNED.coords["position"] = sc.geometry.position(X, Y, Z)


class TestResolution(unittest.TestCase):
    def test_angle_with_gravity(self):
        BINNED.bins.constituents["data"].coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.angstrom,
        )
        z_pixel = 1.0 * sc.units.m
        z_sample = 0.0 * sc.units.m
        y_pixel = 1.0 * sc.units.m
        y_sample = 0.0 * sc.units.m
        x_pixel = 0.0 * sc.units.m
        x_sample = 0.0 * sc.units.m
        pixel_position = sc.geometry.position(x_pixel, y_pixel, z_pixel)
        sample_position = sc.geometry.position(x_sample, y_sample, z_sample)
        actual_result = corrections.angle_with_gravity(
            BINNED, pixel_position, sample_position
        )
        assert_almost_equal(
            actual_result.values,
            [
                45.0000359,
                45.0001436,
                45.000009,
                45.000009,
                45.0001436,
                45.0000359,
                45.0001436,
                45.0000359,
                45.0001436,
            ],
        )

    def test_y_dash0(self):
        velocity = 10.0 * (sc.units.m / sc.units.s)
        z_measured = 1.0 * sc.units.m
        z_origin = 0.0 * sc.units.m
        y_measured = 1.0 * sc.units.m
        y_origin = 0.0 * sc.units.m
        expected_result = 1.04903325
        actual_result = corrections.y_dash0(
            velocity, z_origin, y_origin, z_measured, y_measured
        )
        assert_almost_equal(actual_result.values, expected_result)

    def test_illumination_correction_no_spill(self):
        beam_size = 1.0 * sc.units.m
        sample_size = 10.0 * sc.units.m
        theta = 30.0 * sc.units.deg
        expected_result = 1
        actual_result = corrections.illumination_correction(
            beam_size, sample_size, theta
        )
        assert_almost_equal(actual_result, expected_result)

    def test_illumination_correction_with_spill(self):
        beam_size = 1.0 * sc.units.m
        sample_size = 0.5 * sc.units.m
        theta = 30.0 * sc.units.deg
        expected_result = 0.59490402718695351
        actual_result = corrections.illumination_correction(
            beam_size, sample_size, theta
        )
        assert_almost_equal(actual_result, expected_result)

    def test_illumination_of_sample_big_sample(self):
        beam_size = 1.0 * sc.units.m
        sample_size = 10.0 * sc.units.m
        theta = 90.0 * sc.units.deg
        expected_result = 1.0 * sc.units.m
        actual_result = corrections.illumination_of_sample(
            beam_size, sample_size, theta
        )
        assert_almost_equal(actual_result.values, expected_result.values)

    def test_illumination_of_sample_small_sample(self):
        beam_size = 1.0 * sc.units.m
        sample_size = 0.5 * sc.units.m
        theta = 90.0 * sc.units.deg
        expected_result = 0.5 * sc.units.m
        actual_result = corrections.illumination_of_sample(
            beam_size, sample_size, theta
        )
        assert_almost_equal(actual_result.values, expected_result.values)

    def test_illumination_of_sample_big_sample(self):
        beam_size = 1.0 * sc.units.m
        sample_size = 10.0 * sc.units.m
        theta = 30.0 * sc.units.deg
        expected_result = 2.0 * sc.units.m
        actual_result = corrections.illumination_of_sample(
            beam_size, sample_size, theta
        )
        assert_almost_equal(actual_result.values, expected_result.values)
