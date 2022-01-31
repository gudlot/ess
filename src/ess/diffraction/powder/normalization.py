# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

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


def _common_wavelength_edges(*data):
    """
    The data has wavelength bin edges for each spectrum:

        ^      |---|
    spectrum  |--|
        v        |--|
            < wavelength >

    This function computes common edges for all spectra
    for the combination of all inputs:

        ^     | --- |
    spectrum  |--   |
        v     |   --|
            < wavelength >

    """
    min_wavelength = sc.concat(
        [d.coords['wavelength']['wavelength', 0].min() for d in data],
        '_aux').min('_aux')
    max_wavelength = sc.concat(
        [d.coords['wavelength']['wavelength', 1].max() for d in data],
        '_aux').min('_aux')
    return sc.concat([min_wavelength, max_wavelength], 'wavelength')


def subtract_empty_instrument(data, empty_instr):
    data = data.copy(deep=False)
    empty_instr = empty_instr.copy(deep=False)

    edges = _common_wavelength_edges(data, empty_instr)
    data.coords['wavelength'] = edges
    empty_instr.coords['wavelength'] = edges

    return data.bins.concat(-empty_instr)
