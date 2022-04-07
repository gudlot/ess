# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import scipp as sc


def supermirror_calibration(data_array: sc.DataArray,
                            m_value: sc.Variable = sc.scalar(
                                5, unit=sc.units.dimensionless),
                            critical_edge: sc.Variable = 0.022 * sc.Unit('1/angstrom'),
                            alpha: sc.Variable = sc.scalar(
                                0.25 / 0.088, unit=sc.units.angstrom)) -> sc.Variable:
    """
    Calibrate supermirror measurements

    :param data_array: Data array to get q-bins/values from.
    :param m_value: m-value for the supermirror.
    :param critical_edge: Supermirror critical edge.
    :param alpha: Supermirror alpha value.

    :return: Calibrated supermirror measurement.
    """
    calibration = calibration_factor(data_array, m_value, critical_edge, alpha)
    data_array_cal = data_array * calibration
    try:
        data_array_cal.attrs[
            'orso'].value.reduction.corrections += ['supermirror calibration']
    except KeyError:
        raise UserWarning("For metadata to be logged in the data array, "
                          "it is necessary to install the orsopy package.")
    return data_array_cal


def calibration_factor(data_array: sc.DataArray,
                       m_value: sc.Variable = sc.scalar(
                           5, unit=sc.units.dimensionless),
                       critical_edge: sc.Variable = 0.022 * sc.Unit('1/angstrom'),
                       alpha: sc.Variable = sc.scalar(
                                0.25 / 0.088, unit=sc.units.angstrom)) -> sc.Variable:
    """
    Return the calibration factor for the supermirror.

    :param data_array: Data array to get q-bins/values from.
    :param m_value: m-value for the supermirror.
    :param critical_edge: Supermirror critical edge.
    :param alpha: Supermirror alpha value.

    :return: Calibration factor at the midpoint of each Q-bin.
    """
    q = data_array.coords['Q']
    if data_array.coords.is_edges('Q'):
        q = sc.midpoints(q)
    max_q = m_value * critical_edge
    lim = (q < critical_edge).astype(float)
    lim.unit = 'one'
    nq = 1.0 / (1.0 - alpha * (q - critical_edge))
    calibration_factor = sc.where(q < max_q, lim + (1 - lim) * nq, sc.scalar(1.0))
    return calibration_factor
