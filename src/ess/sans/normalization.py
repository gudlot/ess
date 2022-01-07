# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippneutron as scn


def solid_angle(data, pixel_width, pixel_height):
    """
    Solid angle computed from rectangular pixels with a 'width' and a 'height'.
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


def transmission_fraction(data_incident, data_transmission, direct_incident,
                          direct_transmission):
    """
    Approximation based on equations in CalculateTransmission documentation
    p = \frac{S_T}{D_T}\frac{D_I}{S_I}
    This is equivalent to mantid.CalculateTransmission without fitting
    """

    return (data_transmission / direct_transmission) * (direct_incident / data_incident)
