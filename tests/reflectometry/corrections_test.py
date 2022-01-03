# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Andrew R. McCluskey (arm61)
import scipp as sc
from ess.reflectometry import corrections


def test_illumination_correction_no_spill():
    beam_size = 1.0 * sc.units.m
    sample_size = 10.0 * sc.units.m
    theta = sc.array(values=[30.0], unit=sc.units.deg, dims=['event'])
    expected_result = sc.scalar(1.0)
    actual_result = corrections.illumination_correction(beam_size, sample_size, theta)
    assert sc.allclose(actual_result, expected_result)


def test_illumination_correction_with_spill():
    beam_size = 1.0 * sc.units.m
    sample_size = 0.5 * sc.units.m
    theta = sc.array(values=[30.0], unit=sc.units.deg, dims=['event'])
    expected_result = sc.scalar(0.59490402718695351)
    actual_result = corrections.illumination_correction(beam_size, sample_size, theta)
    assert sc.allclose(actual_result, expected_result)


def test_illumination_of_sample_big_sample():
    beam_size = 1.0 * sc.units.m
    sample_size = 10.0 * sc.units.m
    theta = 90.0 * sc.units.deg
    expected_result = 1.0 * sc.units.m
    actual_result = corrections.illumination_of_sample(beam_size, sample_size, theta)
    assert sc.allclose(actual_result, expected_result)


def test_illumination_of_sample_small_sample():
    beam_size = 1.0 * sc.units.m
    sample_size = 0.5 * sc.units.m
    theta = 90.0 * sc.units.deg
    expected_result = 0.5 * sc.units.m
    actual_result = corrections.illumination_of_sample(beam_size, sample_size, theta)
    assert sc.allclose(actual_result, expected_result)


def test_illumination_of_sample_off_angle():
    beam_size = 1.0 * sc.units.m
    sample_size = 10.0 * sc.units.m
    theta = 30.0 * sc.units.deg
    expected_result = 2.0 * sc.units.m
    actual_result = corrections.illumination_of_sample(beam_size, sample_size, theta)
    assert sc.allclose(actual_result, expected_result)


def test_illumination_range():
    beam_size = 100.0 * sc.units.m
    sample_size = 10.0 * sc.units.m
    theta = sc.array(values=[15.0, 30.0], unit=sc.units.deg, dims=[''])
    expected_result = sc.array(values=[10., 10.], unit=sc.units.m, dims=[''])
    actual_result = corrections.illumination_of_sample(beam_size, sample_size, theta)
    assert sc.allclose(actual_result, expected_result)
