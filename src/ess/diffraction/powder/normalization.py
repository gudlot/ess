# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

import uuid

import scipp as sc
from scippneutron.tof.conversions import beamline, elastic

from .smoothing import fft_smooth
from .tools import unwrap_attr


def normalize_by_monitor(data: sc.DataArray,
                         *,
                         monitor: str,
                         wavelength_edges: sc.Variable,
                         smooth_args=None) -> sc.DataArray:
    """

    """

    mon = unwrap_attr(data.meta[monitor])
    if 'wavelength' not in mon.coords:
        mon = mon.transform_coords('wavelength',
                                   graph={
                                       **beamline(scatter=False),
                                       **elastic("tof")
                                   },
                                   keep_inputs=False,
                                   keep_intermediate=False,
                                   keep_aliases=False)

    mon = sc.rebin(mon, 'wavelength', wavelength_edges)
    if smooth_args is not None:
        print(f"Smoothing monitor '{monitor}' for normalisation with {smooth_args}.")
        mon = fft_smooth(mon, dim='wavelength', **smooth_args)
    return data.bins / sc.lookup(func=mon, dim='wavelength')


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

    edges = _common_edges(data, empty_instr, dim='wavelength')
    data.coords['wavelength'] = edges
    empty_instr.coords['wavelength'] = edges

    return data.bins.concat(-empty_instr)
