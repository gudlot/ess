# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
from typing import Any, Optional, Dict

import scipp as sc
from scippneutron.tof.conversions import beamline, elastic

from .smoothing import lowpass
from ..logging import get_logger


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
        :func:`ess.diffraction.lowpass` before dividing into `data`.
        `smooth_args` is passed as keyword arguments to
        :func:`ess.diffraction.lowpass`. If ``None``, the monitor is not smoothed.

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
            "Smoothing monitor '%s' for normalisation using "
            "ess.diffraction.smoothing.lowpass with %s.", monitor, smooth_args)
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
