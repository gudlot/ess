# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippneutron as scn


def solid_angle_of_rectangular_pixels(data, pixel_width, pixel_height):
    """
    Solid angle computed from rectangular pixels with a 'width' and a 'height'.

    Note that this is an approximation which is only valid for small angles
    between the line of sight and the rectangle normal.
    This is usually a reasonable approximation for SANS experiments.
    """
    L2 = scn.L2(data)
    return (pixel_width * pixel_height) / (L2 * L2)


def subtract_background_and_rebin(data, wavelength_bins, threshold):
    """
    Subtracts background value from data counts and performs a wavelength rebin.
    The background is computed as the mean value of all the counts below the given
    threshold.
    """
    data_original = data.copy(deep=False)
    data_original.masks['background'] = data_original.data > threshold
    background = sc.mean(data_original)
    out = data - background
    out = sc.rebin(out, "wavelength", wavelength_bins)
    return out


def transmission_fraction(data_incident_monitor, data_transmission_monitor,
                          direct_incident_monitor, direct_transmission_monitor):
    """
    Approximation based on equations in CalculateTransmission documentation
    p = \frac{S_T}{D_T}\frac{D_I}{S_I}
    This is equivalent to mantid.CalculateTransmission without fitting.

    TODO: It seems we are always multiplying this by data_incident_monitor to compute
    the normalization term. We could consider just returning
    data_transmission_monitor * direct_incident_monitor / direct_transmission_monitor
    """

    return (data_transmission_monitor / direct_transmission_monitor) * (
        direct_incident_monitor / data_incident_monitor)
