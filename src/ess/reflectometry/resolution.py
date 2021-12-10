# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Andrew R. McCluskey (arm61)

import numpy as np
import scipp as sc


def detector_resolution(spatial_resolution: sc.Variable, pixel_position: sc.Variable,
                        sample_position: sc.Variable) -> sc.Variable:
    """
    Calculate the resolution function due to the spatial resolution of the detector.
    The function returns the standard deviation of detector resolution.

    :param spatial_resolution: Detector spatial resolution.
    :param pixel_position: The position of each pixel in the dimension parallel to the
        beam.
    :param sample_position: The position of the sample in the dimension parallel to the
        beam.
    """
    fwhm = sc.to_unit(sc.atan(spatial_resolution / (pixel_position - sample_position)),
                      "deg")
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def z_offset(position: sc.Variable, offset: sc.Variable) -> sc.Variable:
    """
    Compute new positions that have been offset in the z-dimension.

    :param position: Position vectors (dtype `vector_3_float64`).
    :param offset: Offset to be applied along the z-dimension.
    """
    position = position.copy()
    position.fields.z += offset
    return position
