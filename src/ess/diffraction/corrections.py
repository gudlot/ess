# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
from typing import Any, Optional, Dict
import uuid

import scipp as sc
from scippneutron.tof.conversions import beamline, elastic

from .smoothing import fft_smooth
from ..logging import get_logger


def merge_calibration(*, into: sc.DataArray, calibration: sc.Dataset) -> sc.DataArray:
    """
    Return a :class:`scipp.DataArray` containing calibration metadata.

    :param into: Base data and metadata for the returned object.
    :param calibration: Calibration data.
    :return: (Shallow) Copy of `into` with additional coordinates and masks
             from `calibration`.
    :seealso: :func:`ess.diffraction.load_calibration`
    """
    dim = calibration.dim
    if not sc.identical(into.coords[dim], calibration.coords[dim]):
        raise ValueError(
            f'Coordinate {dim} of calibration and target dataset do not agree.')
    out = into.copy(deep=False)
    for name in ('difa', 'difc', 'tzero'):
        out.attrs[name] = calibration[name].data
    out.masks['cal'] = calibration['mask'].data
    return out


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

    def extremum(fn, index):
        aux_dim = uuid.uuid4().hex
        return fn(sc.concat([fn(edge[dim, index]) for edge in edges], aux_dim), aux_dim)

    lo = extremum(sc.min, 0)
    hi = extremum(sc.max, 1)
    return sc.concat([lo, hi], dim)


def subtract_empty_instrument(data: sc.DataArray,
                              empty_instr: sc.DataArray) -> sc.DataArray:
    """
    Combine event list of data with that of an empty instrument measurement.

    :param data: Binned data in wavelength.
    :param empty_instr: Binned data for an empty instrument in wavelength.
    :return: Binned data containing events from `data` with positive weights and
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
                         wavelength_edges: sc.Variable,
                         smooth_args: Optional[Dict[str, Any]] = None) -> sc.DataArray:
    """
    Normalize event data by a histogrammed monitor.

    :param data: Input data.
    :param monitor: Name of a monitor. Must be stored as metadata in `data`.
    :param wavelength_edges: Histogram the monitor with these edges.
    :param smooth_args: If given, the monitor histogram is smoothed with
                        :func:ess.diffraction.fft_smooth` before dividing into `data`.
                        `smooth_args` is passed as arguments to `fft_smooth`.
                        If ``None``, the monitor is not smoothed.
    :return: `data` normalized by a monitor.
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

    mon = sc.rebin(mon, 'wavelength', wavelength_edges)
    if smooth_args is not None:
        get_logger('diffraction').info(
            "Smoothing monitor '%s' for normalisation using fft_smooth with %s.",
            monitor, smooth_args)
        mon = fft_smooth(mon, dim='wavelength', **smooth_args)
    return data.bins / sc.lookup(func=mon, dim='wavelength')


def normalize_by_proton_charge(data: sc.DataArray,
                               *,
                               proton_charge: str = 'gd_prtn_chrg',
                               in_place: bool = False) -> sc.DataArray:
    proton_charge = data.attrs[proton_charge]
    if in_place:
        data /= proton_charge
        return data
    return data / proton_charge


def normalize_by_vanadium(data: sc.DataArray,
                          *,
                          vanadium: sc.DataArray,
                          edges: sc.Variable,
                          in_place: bool = False) -> sc.DataArray:
    """
    Normalize sample data by a vanadium measurement.

    :param data: Sample data.
    :param vanadium: Vanadium data.
    :param edges: Histogram `vanadium` with these bin edges.
    :param in_place: If ``True``, `data` is modified in order to safe memory.
                     Otherwise, the input data is unchanged.
    :return: `data` normalized by `vanadium`. This is the same object as `data`
             when ``in_place == True``.
    """
    norm = sc.lookup(sc.histogram(vanadium, bins=edges), dim=edges.dim)
    if in_place:
        data.bins /= norm
        out = data
    else:
        out = data.bins / norm
    return sc.histogram(out, bins=edges)
