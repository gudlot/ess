# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# flake8: noqa: E501
"""
Code for the quantification of resolution effects in reflectometry measurements.
"""

# author: Andrew R. McCluskey (arm61)

import numpy as np
import scipp as sc


def detector_resolution(spatial_resolution, pixel_position, sample_position):
    """
    Calculation the resolution function due to the spatial resolution of the detector.

    Args:
        spatial_resolution (:py:class:`scipp._scipp.core.Variable`): Detector spatial resolution.
        pixel_position (:py:class:`scipp._scipp.core.Variable`): The position of each pixel in the dimension parallel to the beam.
        sample_position (:py:class:`scipp._scipp.core.Variable`): The position of the sample in the dimension parallel to the beam.

    Returns:
        (:py:class:`scipp._scipp.core.Variable`): Standard deviation of detector resolution.
    """
    fwhm = sc.to_unit(
        sc.atan(spatial_resolution / (pixel_position - sample_position)),
        "deg",
    )
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def z_offset(position, offset_value):
    """
    Offset in the z-dimension.

    Args:
        position (:py:class:`scipp._scipp.core.Variable`): Position variable, should be :code:`vector_3_float64`.
        offset_value (:py:class:`scipp._scipp.core.Variable`): z-dimension offset value.

    Returns:
        (:py:class:`scipp._scipp.core.Variable`): New position variables that has been offset.
    """
    return sc.geometry.position(position.x1, position.x2,
                                position.x3 + offset_value)
