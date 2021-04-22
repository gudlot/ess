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
        spatial_resolution (`sc.Variable`): Detector spatial resolution.
        pixel_position (`sc.Variable`): The position of each pixel in the dimension parallel to the beam.
        sample_position (`sc.Variable`): The position of the sample in the dimension parallel to the beam.

    Returns:
        (`sc.Variable`): Standard deviation of detector resolution.
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
        position (`sc.Variable`): Position variable, should be `dtype=vector_3_float64`.
        offset_value (`sc.Variable`): z-dimension offset value.

    Returns:
        (`sc.Variable`): New position variables that has been offset.
    """
    return sc.geometry.position(
        sc.geometry.x(position),
        sc.geometry.y(position),
        sc.geometry.z(position) + offset_value,
    )
