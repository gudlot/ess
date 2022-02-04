# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
from scipp.signal import butter
import scipp as sc

from ...logging import get_logger


def _ensure_no_variances(var: sc.Variable) -> sc.Variable:
    if var.variances is not None:
        get_logger('diffraction').warning(
            'Tried to smoothen data with uncertainties. '
            'This is not supported because the results would be highly correlated.\n'
            'Instead, the variances are ignored and the output '
            'will be returned without any!'
            '\n--------------------------------------------------\n'
            'If you know a good solution for handling uncertainties in such a case, '
            'please contact the scipp developers! (e.g. via https://github.com/scipp)'
            '\n--------------------------------------------------\n')
        return sc.values(var)
    return var


def fft_smooth(var: sc.Variable, *, dim: str, order: int,
               Wn: sc.Variable) -> sc.Variable:
    var = _ensure_no_variances(var)

    if var.coords[dim].sizes[dim] == var.sizes[dim] + 1:
        # TODO allow dim in attrs
        var = var.copy(deep=False)
        var.coords[dim] = sc.midpoints(var.coords[dim], dim)

    return butter(var.coords[dim], N=order, Wn=Wn).filtfilt(var, dim)
