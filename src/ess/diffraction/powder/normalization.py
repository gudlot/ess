# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

import scipp as sc
from scippneutron.tof.conversions import beamline, elastic

from .smoothing import smooth_data
from .tools import unwrap_attr


def normalize_by_monitor(data: sc.DataArray, *, monitor: str,
                         wavelength_edges: sc.Variable) -> sc.DataArray:
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
    mon = smooth_data(mon, dim='wavelength', NPoints=40)

    mon = sc.rebin(mon, 'wavelength', wavelength_edges)
    # TODO why does this produce different values?
    #   How do we know what is the correct binning?
    # mi = wavelength_edges.min()
    # ma = wavelength_edges.max()
    # mon = mon['wavelength', mi:ma]

    return data.bins / sc.lookup(func=mon, dim='wavelength')
