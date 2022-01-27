# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

from scipp.signal import butter
import scipp as sc


def smooth_data(variable, *, dim, NPoints=3):
    """
    Function that smooths data by assigning the value of each point to the
    mean of the values from surrounding points and itself. The number of points
    to average is NPoint, which is odd so that the point itself is in the
    center of the averaged interval. If an even NPoints is given, it is
    incremented. At the ends of the interval the full number of points is not
    used, but all available within NPoints//2 is.

    Parameters
    ----------
        variable: scipp variable
            The variable which should have its values smoothed

        dim: scipp Dim
            The dimension along which values should be smoothed

        NPoints: int
            The number of points to use in the mean (odd number)
    """

    if NPoints < 3:
        raise ValueError("smooth_data needs NPoints of 3 or higher.")

    if variable.variances is not None:
        # TODO log properly
        print('WARNING ignoring variances')
        variable = sc.values(variable)

    data_length = variable.sizes[dim]
    out = variable.copy()  # preallocate output variable

    hr = NPoints // 2  # half range rounded down

    for index in range(data_length):
        begin = max(0, index - hr)
        end = min(data_length, index + hr + 1)
        out[dim, index] = sc.mean(variable[dim, begin:end], dim)

    return out


# TODO remove when scipp finally supports this
def _to_bin_centers(coord):
    return coord[:-1] + (coord[1:] - coord[:-1]) / 2


def fft_smooth(data, *, dim, order, Wn):
    if data.variances is not None:
        # TODO log properly
        print('WARNING ignoring variances')
        data = sc.values(data)

    if data.coords[dim].sizes[dim] == data.sizes[dim] + 1:
        # TODO allow dim in attrs
        data = data.copy(deep=False)
        data.coords[dim] = _to_bin_centers(data.coords[dim])

    return butter(data.coords[dim], N=order, Wn=Wn).filtfilt(data, dim)
