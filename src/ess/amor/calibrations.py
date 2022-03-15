# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import scipp as sc


def supermirror_calibration(
        coord: sc.Variable,
        is_bin_edge: bool = True,
        m_value: sc.Variable = sc.scalar(5, unit=sc.units.dimensionless),
        critical_edge: sc.Variable = 0.022 * sc.Unit('1/angstrom'),
        alpha: sc.Variable = 0.25 / 0.088 * sc.units.angstrom) -> sc.Variable:
    """
    Determine calibration factor for the supermirror.

    :param coord: Q-bins or centres over which calibration is performed.
    :param is_bin_edge: Defines if the coord is a bin edge.
    :param m_value: m-value for the supermirror.
    :param critical_edge: Supermirror critical edge.
    :param alpha: Supermirror alpha value.

    :return: Calibration factor at the midpoint of each Q-bin.
    """
    if is_bin_edge:
        q = sc.midpoints(coord, 'Q')
    else:
        q = coord
    max_q = m_value * critical_edge
    lim = (q < critical_edge).astype(float)
    lim.unit = 'one'
    nq = 1.0 / (1.0 - alpha * (q - critical_edge))
    calibration_factor = sc.where(q < max_q, lim + (1 - lim) * nq, sc.scalar(1.0))    
    return calibration_factor

