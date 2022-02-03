# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import scipp as sc


def merge_calibration(*, into: sc.DataArray, calibration: sc.Dataset) -> sc.DataArray:
    dim = calibration.dim
    if not sc.identical(into.coords[dim], calibration.coords[dim]):
        raise ValueError(
            f'Coordinate {dim} of calibration and target dataset do not agree.')
    out = into.copy(deep=False)
    for name in ('difa', 'difc', 'tzero'):
        out.attrs[name] = calibration[name].data
    out.masks['cal'] = calibration['mask'].data
    return out
