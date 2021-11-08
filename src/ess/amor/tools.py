# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Andrew R. McCluskey (arm61)

import scipp as sc


def q_grid(q_min: sc.Variable = 0.008 * sc.Unit('1/angstrom'),
           q_fix: sc.Variable = 0.005 * sc.Unit('1/angstrom'),
           q_max: sc.Variable = 0.08 * sc.Unit('1/angstrom'),
           d_q: sc.Variable = None) -> sc.Variable:
    """
    Generate bin edges for a `q_grid` based on the custom linear-log grid from
    Jochen Stahn at PSI. The units of the bins are reciprocal angstrom.

    :param q_min: The minimum q-value to be present in the data.
        Default is `0.008 Å^{-1}`.
    :param q_fix: The last point in the linear binning and first in the log binning.
        Default is `0.005 Å^{-1}`.
    :param q_max: The maximum q-value to be present in the data.
        Default is `0.08 Å^{-1}`.
    :param q_d: The q-spacing for the linear region, from which the log-spacing is
        defined. Default is `0.05*q_fix Å^{-1}`.
    """
    q_min = sc.to_unit(q_min, q_max.unit)
    q_fix = sc.to_unit(q_fix, q_max.unit)
    if d_q is None:
        d_q = 0.05 * q_fix
    if (q_min < q_fix).value and (q_fix < q_max).value:
        n_linear = ((q_fix - q_min) / d_q + 0.5).astype(sc.dtype.int64)
        q_linear = q_fix - d_q * sc.arange('qz', n_linear.value, 0, -1)
        n_log = (sc.log(q_max / q_fix) / sc.log(1. + d_q / q_fix) + 0.5).astype(
            sc.dtype.int64)
        q_log = sc.Variable(
            dims=['qz'],
            values=(q_fix.value *
                    (1. + d_q / q_fix).value**sc.arange('qz', 0,
                                                        (n_log + 1).values, 1).values),
            unit=q_max.unit)
        return sc.concatenate(q_linear, q_log, 'qz')
    elif (q_min < q_max).value and (q_max < q_fix).value:
        n_linear = ((q_fix - q_min) / d_q + 0.5).astype(sc.dtype.int64)
        m_linear = ((q_fix - q_max) / d_q - 0.5).astype(sc.dtype.int64)
        q_linear = q_fix - d_q * sc.arange('qz', n_linear.value,
                                           (m_linear - 1).value, -1)
        return q_linear
    elif (q_fix < q_min).value and (q_min < q_max).value:
        n_log = (sc.log(q_max / q_fix) / sc.log(1. + d_q / q_fix) + 0.5).astype(
            sc.dtype.int64)
        m_log = (sc.log(q_min / q_fix) / sc.log(1. + d_q / q_fix) - 0.5).astype(
            sc.dtype.int64)
        q_log = sc.Variable(
            dims=['qz'],
            values=q_fix.value *
            (1. + d_q / q_fix).values**sc.arange('qz', m_log.value,
                                                 (n_log + 1).value, 1.).values,
            unit=q_max.unit)
        return q_log
    else:
        raise ValueError("The given values of input parameters are not compatible.")
