# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
from typing import Optional

import scipp as sc

from .corrections import merge_calibration


def dspacing_from_diff_calibration(tof: sc.Variable, tzero: sc.Variable,
                                   difa: sc.Variable, difc: sc.Variable) -> sc.Variable:
    # TODO Use of where is inefficient, move to C++
    # DIFA non zero: tof = DIFA * d**2 + DIFC * d + TZERO.
    # d-spacing is the positive solution of this polynomial
    return sc.where(difa == sc.scalar(0.0, unit=difa.unit),
                    sc.reciprocal(difc) * (tof - tzero),
                    (sc.sqrt(difc**2 + 4 * difa * (tof - tzero)) - difc) / (2.0 * difa))


def _restore_tof_if_in_wavelength(data: sc.DataArray) -> sc.DataArray:
    if 'wavelength' not in data.dims:
        return data

    # TODO better error message
    # TODO remove empty graph
    tof_data = data.transform_coords('tof', graph={'_': '_'})
    del tof_data.coords['wavelength']
    if tof_data.bins:
        del tof_data.bins.coords['wavelength']
    return tof_data.rename_dims({'wavelength': 'tof'})


def to_dspacing_with_calibration(
        data: sc.DataArray,
        *,
        calibration: Optional[sc.Dataset] = None) -> sc.DataArray:
    if calibration is not None:
        out = merge_calibration(into=data, calibration=calibration)
    else:
        out = data

    out = _restore_tof_if_in_wavelength(out)
    graph = {'dspacing': dspacing_from_diff_calibration}
    out = out.transform_coords('dspacing', graph=graph)

    for name in ('sample_position', 'source_position', 'position'):
        if name in out.coords:
            out.attrs[name] = out.coords.pop(name)
        if out.bins is not None and name in out.bins.coords:
            out.bins.attrs[name] = out.bins.coords.pop(name)

    return out
