# flake8: noqa: E501
"""
This module is focused on enabling different binning for reflectometry data.
"""

# author: Andrew R. McCluskey (arm61)

import numpy as np
import scipp as sc
import scippneutron as scn


def q_bin(data, bins=None, unit=None):
    """
    Return data that has been binned in the q-bins passed.

    Args:
        data (`ess.reflectometry.ReflData` or `ess.amor.AmorData` or `ess.amor.AmorReference`): data to be binned.
        bins (`array_like`, optional): q-bin edges. Defaults to 200 bins between max and min q.
        unit (`scipp._scipp.core.Unit`, optional): Unit of q bins. Defaults to 1/Å.

    Returns:
        (`scipp._scipp.core.DataArray`): Data array binned into qz with resolution.
    """
    if "qz" in data.event.coords and "tof" in data.event.coords:
        if unit is None:
            unit = data.event.coords['qz'].unit
        erase = ['tof'] + data.data.dims
        q_z_vector = sc.to_unit(data.event.coords["qz"], unit)
        if bins is None:
            bins = np.linspace(
                sc.min(q_z_vector).values,
                sc.max(q_z_vector).values, 200)
        data.event.coords["qz"] = q_z_vector
        edges = sc.array(dims=["qz"], unit=unit, values=bins)
        binned = sc.bin(data.data, erase=erase, edges=[edges])
        if "sigma_qz_by_qz" in data.event.coords:
            qzr = []
            for i in binned.data.values:
                try:
                    qzr.append(i.coords["sigma_qz_by_qz"].values.max())
                except ValueError:
                    qzr.append(0)
            qzr = np.array(qzr)
            binned.coords["sigma_qz_by_qz"] = sc.Variable(values=qzr,
                                                          dims=["qz"])
    else:
        raise sc.NotFoundError("qz or tof coordinate cannot be found.")
    return binned / (data.event.shape[0] * sc.units.dimensionless)


def two_dimensional_bin(data, dims, bins=None, units=None):
    """
    Perform some arbitrary two-dimensional binning.

    Args:
        data (`ess.reflectometry.ReflData` or `ess.amor.AmorData` or `ess.amor.AmorReference`): data to be binned.
        dims (`tuple` of `str`): The dimensions to be binned
        bins (`tuple` of `array_like`, optional): Bin edges. Optional, defaults to min and max with 50 bins in each dim.
        unit (`scipp._scipp.core.Unit`): Unit of bins. Optional, defaults to units of coord.

    Returns:
        (`scipp._scipp.core.DataArray`): Data array binned into qz with resolution.
    """
    for d in dims:
        if d not in list(data.event.coords):
            raise sc.NotFoundError(f'dim {d} not found in coords')
    if bins is None:
        bins = []
        for d in dims:
            vals = data.event.coords[d].values
            bins.append(np.linspace(vals.min(), vals.max(), 50))
    if units is None:
        units = []
        for d in dims:
            units.append(data.event.coords[d].unit)
    bin_edges = []
    for i, d in enumerate(dims):
        unit_change = sc.to_unit(data.event.coords[d], units[i])
        data.event.coords[d] = unit_change
        bin_edges.append(sc.array(dims=[d], unit=units[i], values=bins[i]))
    return sc.bin(data.data.bins.concatenate('detector_id'), edges=bin_edges)
