# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
from typing import Any, Optional, Dict

import scipp as sc
from scippneutron.tof.conversions import beamline, elastic

from .smoothing import lowpass
from ..logging import get_logger


def _common_edges(*edges, dim: str) -> sc.Variable:
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
    lo = sc.reduce([e[dim, 0] for e in edges]).min().min()
    hi = sc.reduce([e[dim, -1] for e in edges]).max().max()
    return sc.concat([lo, hi], dim)


def subtract_empty_instrument(data: sc.DataArray,
                              empty_instr: sc.DataArray) -> sc.DataArray:
    """
    Combine event list of data with that of an empty instrument measurement.

    Parameters
    ----------
    data:
        Binned data in wavelength.
    empty_instr:
        Binned data for an empty instrument in wavelength.

    Returns
    -------
    :
        Binned data containing events from `data` with positive weights and
        events from `empty_string` with negative weights.
    """
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


def normalize_by_monitor(data: sc.DataArray,
                         *,
                         monitor: str,
                         wavelength_edges: Optional[sc.Variable] = None,
                         smooth_args: Optional[Dict[str, Any]] = None) -> sc.DataArray:
    """
    Normalize event data by a monitor.

    Parameters
    ----------
    data:
        Input event data.
    monitor:
        Name of a histogrammed monitor. Must be stored as metadata in `data`.
    wavelength_edges:
        If given, rebin the monitor with these edges.
    smooth_args:
        If given, the monitor histogram is smoothed with
        :func:`ess.diffraction.fft_smooth` before dividing into `data`.
        `smooth_args` is passed as keyword arguments to
        :func:`ess.diffraction.fft_smooth`. If ``None``, the monitor is not smoothed.

    Returns
    -------
    :
        `data` normalized by a monitor.
    """
    mon = data.meta[monitor].value
    if 'wavelength' not in mon.coords:
        mon = mon.transform_coords('wavelength',
                                   graph={
                                       **beamline(scatter=False),
                                       **elastic("tof")
                                   },
                                   keep_inputs=False,
                                   keep_intermediate=False,
                                   keep_aliases=False)

    if wavelength_edges is not None:
        mon = sc.rebin(mon, 'wavelength', wavelength_edges)
    if smooth_args is not None:
        get_logger('diffraction').info(
            "Smoothing monitor '%s' for normalisation using fft_smooth with %s.",
            monitor, smooth_args)
        mon = lowpass(mon, dim='wavelength', **smooth_args)
    return data.bins / sc.lookup(func=mon, dim='wavelength')


def normalize_by_vanadium(data: sc.DataArray,
                          *,
                          vanadium: sc.DataArray,
                          edges: sc.Variable,
                          in_place: bool = False) -> sc.DataArray:
    """
    Normalize sample data by a vanadium measurement.

    Parameters
    ----------
    data:
        Sample data.
    vanadium:
        Vanadium data.
    edges:
        `vanadium` is histogrammed into these bins before dividing the data by it.
    in_place:
        If ``True``, `data` is modified in order to safe memory.
        Otherwise, the input data is unchanged.

    Returns
    -------
    :
        `data` normalized by `vanadium`.
    """
    norm = sc.lookup(sc.histogram(vanadium, bins=edges), dim=edges.dim)
    if in_place:
        data.bins /= norm
        return data
    return data.bins / norm
