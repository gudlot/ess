"""
Code for the quantification of resolution effects in reflectometry measurements.
"""

# author: Andrew R. McCluskey (arm61)

import numpy as np
import scipp as sc


def detector_resolution(spatial_resolution, z_pixel_position,
                        z_sample_position):
    """
    Calculation the resolution function due to the spatial resolution of the detector.

    Args:
        spatial_resolution (`sc.Variable`): Detector spatial resolution.
        z_pixel_position (`sc.Variable`): The position of each pixel in the z-dimension.
        z_sample_position (`sc.Variable`): The position of the sample in the z-dimension.

    Returns:
        (`sc.Variable`): Standard deviation of detector resolution.
    """
    fwhm = sc.to_unit(
        sc.atan(spatial_resolution / (z_pixel_position - z_sample_position)),
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
