# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.interpolate import _midpoints


def supermirror_calibration(
        q_bins: sc.Variable,
        m_value: sc.Variable = 5 * sc.Unit('dimensionless'),
        critical_edge: sc.Variable = 0.022 * sc.Unit('1/angstrom'),
        alpha: sc.Variable = 0.25 / 0.088 * sc.units.angstrom) -> sc.Variable:
    """
    Determine calibration factor for the supermirror.

    :param q_bins: Q-bins over which calibration is performed.
    :param m_value: m-value for the supermirror.
    :param critical_edge: Supermirror critical edge.
    :param alpha: Supermirror alpha value.

    :return: Calibration factor at the midpoint of each Q-bin.
    """
    q_midpoints = _midpoints(q_bins, 'Q')
    max_q = m_value * critical_edge
    lim = (q_midpoints < critical_edge).astype(float)
    nq = 1.0 / (1.0 - alpha * (q_midpoints - critical_edge))
    calibration_factor = lim + (1 - lim) * nq
    calibration_factor *= (q_midpoints < max_q).astype(float)
    return calibration_factor
