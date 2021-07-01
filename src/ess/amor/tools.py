# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# flake8: noqa: E501
"""
Tools to help with Amor data reduction.
"""

# author: Andrew R. McCluskey (arm61)

import scipp as sc


def q_grid(q_min=0.008 * sc.Unit('1/angstrom'),
           q_fix=0.005 * sc.Unit('1/angstrom'),
           q_max=0.08 * sc.Unit('1/angstrom'),
           d_q=None):
    """
    Obtain a q_grid based on the custom linear-log grid from Jochen Stahn at PSI. The units of the bins are reciprocal angstrom.

    :param q_min: The minimum q-value to be present in the data. Optional, defaults to 0.008 Å^{-1}
    :type q_min: scipp._scipp.core.Variable
    :param q_fix: The last point in the linear binning and first in the log binning. Optional, defaults to 0.005 Å^{-1}
    :type: q_fix: scipp._scipp.core.Variable
    :param q_max: The maximum q-value to be present in the data. Optional, defaults to 0.08 Å^{-1} 
    :type q_max: scipp._scipp.core.Variable
    :param q_d: The q-spacing for the linear region, from which the log-spacing is defined. Optional, defaults to 0.05*q_fix Å^{-1}
    :type q_d: scipp._scipp.core.Variable

    :return: The bin edges to be used in q-binning.
    :rtype: scipp._scipp.core.Variable 
    """
    if d_q is None:
        d_q = 0.05 * q_fix
    if (q_min < q_fix).value and (q_fix < q_max).value:
        n_linear = ((q_fix - q_min) / d_q + 0.5).astype(sc.dtype.int64)
        q_linear = q_fix - d_q * sc.arange('qz', n_linear.value, 0, -1)
        n_log = (sc.log(q_max / q_fix) / sc.log(1. + d_q / q_fix) +
                 0.5).astype(sc.dtype.int64)
        q_log = sc.Variable(['qz'],
                            values=(q_fix.value * (1. + d_q / q_fix).value**
                                    sc.arange('qz', 0,
                                              (n_log + 1).values, 1).values),
                            unit=sc.Unit('1/angstrom'))
        return sc.concatenate(q_linear, q_log, 'qz')
    elif (q_min < q_max).value and (q_max < q_fix).value:
        n_linear = ((q_fix - q_min) / d_q + 0.5).astype(sc.dtype.int64)
        m_linear = ((q_fix - q_max) / d_q - 0.5).astype(sc.dtype.int64)
        q_linear = q_fix - d_q * sc.arange('qz', n_linear.value,
                                           (m_linear - 1).value, -1)
        return q_linear
    elif (q_fix < q_min).value and (q_min < q_max).value:
        n_log = (sc.log(q_max / q_fix) / sc.log(1. + d_q / q_fix) +
                 0.5).astype(sc.dtype.int64)
        m_log = (sc.log(q_min / q_fix) / sc.log(1. + d_q / q_fix) -
                 0.5).astype(sc.dtype.int64)
        q_log = sc.Variable(
            ['qz'],
            values=q_fix.value *
            (1. + d_q / q_fix).values**sc.arange('qz', m_log.value,
                                                 (n_log + 1).value, 1.).values,
            unit=sc.Unit('1/angstrom'))
        return q_log
    else:
        raise ValueError(
            "The given values of input parameters are not compatible.")
