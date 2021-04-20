# flake8: noqa: E501
"""
This module is focused on enabling different binning for reflectometry data.
"""

# author: Andrew R. McCluskey (arm61)

import numpy as np
import scipp as sc
import scippneutron as scn


def _q_bin(data, bins=None, unit=(1 / sc.units.angstrom).unit):
    """
    Return data that has been binned in the q-bins passed.

    Args:
        data (`ess.reflectometry.ReflData` or `ess.amor.AmorData` or `ess.amor.AmorReference`): data to be binned.
        bins (`array_like`): q-bin edges.
        unit (`scipp._scipp.core.Unit`): Unit for q bins. Defaults to 1/Å.

    Returns:
        (`scipp._scipp.core.DataArray`): Data array binned into qz with resolution.
    """
    if "qz" in data.event.coords and "tof" in data.event.coords:
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


#def two_dimensional_bin(data, dimensions, bins=None, units=[None, None])
