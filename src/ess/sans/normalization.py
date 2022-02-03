# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippneutron as scn


def solid_angle_of_rectangular_pixels(data: sc.DataArray, pixel_width: sc.Variable,
                                      pixel_height: sc.Variable) -> sc.Variable:
    """
    Solid angle computed from rectangular pixels with a 'width' and a 'height'.

    Note that this is an approximation which is only valid for small angles
    between the line of sight and the rectangle normal.

    :param data: The DataArray that contains the positions for the detector pixels and
        the sample.
    :param pixel_width: The width of the rectangular pixels.
    :param pixel_height: The height of the rectangular pixels.
    """
    L2 = scn.L2(data)
    return (pixel_width * pixel_height) / (L2 * L2)


def subtract_background_and_rebin(
        data: sc.DataArray,
        wavelength_bins: sc.Variable,
        non_background_range: sc.Variable = None) -> sc.DataArray:
    """
    Subtracts background value from data counts and performs a wavelength rebin.
    The background is computed as the mean value of all the counts outside of the given
    ``non_background_range``.

    :param data: The DataArray containing the monitor data to be de-noised and rebinned.
    :param wavelength_bins: The wavelength binning to apply when rebinning the data.
    :param non_background_range: The range of wavelengths that defines the data which
        does not constitute background. Everything outside this range is treated as
        background counts.
    """
    if non_background_range is not None:
        data_original = data.copy(deep=False)
        wav_midpoints = 0.5 * (data_original.coords['wavelength'][1:] +
                               data_original.coords['wavelength'][:-1])
        data_original.masks['non_background'] = (
            wav_midpoints > non_background_range[0]) & (wav_midpoints <
                                                        non_background_range[1])
        background = sc.mean(data_original)
        data = data - background
    return sc.rebin(data, "wavelength", wavelength_bins)


def transmission_fraction(data_incident_monitor: sc.DataArray,
                          data_transmission_monitor: sc.DataArray,
                          direct_incident_monitor: sc.DataArray,
                          direct_transmission_monitor: sc.DataArray) -> sc.DataArray:
    """
    Approximation based on equations in CalculateTransmission documentation
    p = \frac{S_T}{D_T}\frac{D_I}{S_I}
    This is equivalent to mantid.CalculateTransmission without fitting.

    TODO: It seems we are always multiplying this by data_incident_monitor to compute
    the normalization term. We could consider just returning
    data_transmission_monitor * direct_incident_monitor / direct_transmission_monitor

    :param data_incident_monitor: The DataArray containing the incident monitor counts
        as a function of wavelength for the data/sample run.
    :param data_transmission_monitor: The DataArray containing the transmission monitor
        counts as a function of wavelength for the data/sample run.
    :param direct_incident_monitor: The DataArray containing the incident monitor counts
        as a function of wavelength for the direct run.
    :param direct_transmission_monitor: The DataArray containing the transmission
        monitor counts as a function of wavelength for the direct run.
    """

    return (data_transmission_monitor / direct_transmission_monitor) * (
        direct_incident_monitor / data_incident_monitor)
