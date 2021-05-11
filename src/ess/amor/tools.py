# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# flake8: noqa: E501
"""
Tools to help with Amor data reduction.
"""

# author: Andrew R. McCluskey (arm61)

import numpy as np


def q_grid(q_min=0.008, q_fix=0.005, q_max=0.08, d_q=None):
    """
    Obtain a q_grid based on the custom linear-log grid from Jochen Stahn at PSI. The units of the bins are reciprocal angstrom.

    Args:
        q_min (:py:attr:`float`, optional): The minimum q-value to be present in the data. Optional, defaults to 0.008 Å^{-1}.
        q_max (:py:attr:`float`, optional): The maximum q-value to be present in the data. Optional, defaults to 0.08 Å^{-1}.
        q_fix (:py:attr:`float`, optional): The last point in the linear binning and first in the log binning. Optional, defaults to 0.005 Å^{-1}.
        q_d (:py:attr:`float`, optional): The q-spacing for the linear region, from which the log-spacing is defined. Optional, defaults to 0.05*q_fix Å^{-1}.

    Returns:
        (:py:attr:`array_like`): The bin edges to be used in q-binning.
    """
    if d_q is None:
        d_q = 0.05 * q_fix
    if q_min < q_fix and q_fix < q_max:
        n_linear = int((q_fix - q_min) / d_q + 0.5)
        q_linear = q_fix - d_q * np.arange(n_linear, 0, -1)
        n_log = int(np.log(q_max / q_fix) / np.log(1. + d_q / q_fix) + 0.5)
        q_log = q_fix * (1. + d_q / q_fix)**np.arange(n_log + 1)
        return np.concatenate((q_linear, q_log))
    elif q_min < q_max and q_max < q_fix:
        n_linear = int((q_fix - q_min) / d_q + 0.5)
        m_linear = int((q_fix - q_max) / d_q - 0.5)
        q_linear = q_fix - d_q * np.arange(n_linear, m_linear - 1, -1)
        return q_linear
    elif q_fix < q_min and q_min < q_max:
        n_log = int(np.log(q_max / q_fix) / np.log(1. + d_q / q_fix) + 0.5)
        m_log = int(np.log(q_min / q_fix) / np.log(1. + d_q / q_fix) - 0.5)
        q_log = q_fix * (1. + d_q / q_fix)**np.arange(m_log, n_log + 1, 1.)
        return q_log
    else:
        raise ValueError(
            "The given values of input parameters are not compatible.")
