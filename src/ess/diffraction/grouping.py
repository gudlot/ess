# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import scipp as sc
import scippneutron as scn


def focus_by_two_theta(data: sc.DataArray,
                       *,
                       edges: sc.Variable,
                       replace_dim: str = 'spectrum') -> sc.DataArray:
    """
    Group data into two_theta bins.

    Parameters
    ----------
    data:
        Input data array with events. Must contain a coord or attr called 'two_theta'
        or coords or attrs that can be used to compute it.
    edges:
        Bin edges in two_theta. `data` is grouped into those bins.
    replace_dim:
        Dimension that is replaced by two_theta.
        All events are concatenated along this dimension.

    Returns
    -------
    :
        `data` grouped into two_theta bins.
    """
    data = data.copy(deep=False)
    if 'two_theta' not in data.meta and 'two_theta' not in data.bins.meta:
        data.coords['two_theta'] = scn.two_theta(data)
    return sc.groupby(data,
                      'two_theta',
                      bins=edges.to(unit=data.coords['two_theta'].unit,
                                    copy=False)).bins.concat(replace_dim)
