# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import uuid

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


def _common_edges(*edges, dim):
    """
    The data has separate bin edges for each spectrum:

        ^      |---|
    spectrum  |--|
        v        |--|
              < dim >

    This function computes common edges for all spectra
    for the combination of all inputs:

        ^     | --- |
    spectrum  |--   |
        v     |   --|
              < dim >
    """

    def extremum(fn, index):
        aux_dim = uuid.uuid4().hex
        return fn(sc.concat([fn(edge[dim, index]) for edge in edges], aux_dim), aux_dim)

    lo = extremum(sc.min, 0)
    hi = extremum(sc.max, 1)
    return sc.concat([lo, hi], dim)


def subtract_empty_instrument(data, empty_instr):
    data = data.copy(deep=False)
    empty_instr = empty_instr.copy(deep=False)

    wavelength_edges = _common_edges(data.coords['wavelength'],
                                     empty_instr.coords['wavelength'],
                                     dim='wavelength')
    data.coords['wavelength'] = wavelength_edges
    empty_instr.coords['wavelength'] = wavelength_edges

    if 'tof' in data.attrs:
        tof_edges = _common_edges(data.attrs['tof'],
                                  empty_instr.attrs['tof'],
                                  dim='wavelength')
        data.attrs['tof'] = tof_edges
        empty_instr.attrs['tof'] = tof_edges

    return data.bins.concat(-empty_instr)
