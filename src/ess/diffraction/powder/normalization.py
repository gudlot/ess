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
        print(f'Smoothing monitor {monitor} for normalisation with {smooth_args}.')
        mon = fft_smooth(mon, dim='wavelength', **smooth_args)
    return data.bins / sc.lookup(func=mon, dim='wavelength')
