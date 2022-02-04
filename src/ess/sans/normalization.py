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


def compute_denominator(direct_beam: sc.DataArray, data_incident_monitor: sc.DataArray,
                        transmission_fraction: sc.DataArray,
                        solid_angle: sc.Variable) -> sc.DataArray:
    """
    Compute the denominator term.
    Because we are histogramming the Q values of the denominator further down in the
    workflow, we convert the wavelength coordinate of the denominator from bin edges to
    bin centers.
    """
    denominator = (solid_angle * direct_beam * data_incident_monitor *
                   transmission_fraction)
    # TODO: once scipp-0.12 is released, use sc.midpoints()
    denominator.coords['wavelength'] = 0.5 * (denominator.coords['wavelength'][1:] +
                                              denominator.coords['wavelength'][:-1])
    return denominator


def normalize(numerator: sc.DataArray, denominator: sc.DataArray) -> sc.DataArray:
    """
    Perform normalization of counts as a fucntion of Q.
    If the numerator contains events, we use the sc.lookup function to perform the
    division.

    :param numerator: The data whose counts will be divided by the denominator. This
        can either be event or dense (histogrammed) data.
    :param denominator: The divisor for the normalization operation. This cannot be
        event data, it must contain histogrammed data.
    """
    if numerator.bins is not None:
        return numerator.bins / sc.lookup(func=denominator, dim='Q')
    else:
        return numerator / denominator
