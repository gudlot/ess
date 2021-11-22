# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Andrew R. McCluskey (arm61)

import scipp as sc
from typing import Union


def q_grid(edges: sc.Variable, scale: Union[list, str], num: Union[list,
                                                                   int]) -> sc.Variable:
    """
    Generate bin edges for a `q_grid` inspired by the custom linear-log grid from
    Jochen Stahn at PSI.

    :param edges: The edges for the different parts of the mesh.
    :param scale: A string or list of strings specifying the scaling for the different
        parts of the mesh. Possible values for the scaling are `"linear"` and `"log"`.
        If a list is supplied, the length of the list must be one less than the length
        of the `edges` parameter.
    :param num: An integer or a list of integers specifying the number of points to use
        in each part of the mesh. If a list is supplied, the length of the list must be
        one less than the length of the `edges` parameter.
    """
    if not isinstance(scale, list):
        scale = [scale]
    if not isinstance(num, list):
        num = [num]
    if len(scale) != len(edges) - 1:
        raise ValueError("Sizes do not match")

    funcs = {"linear": sc.linspace, "log": sc.geomspace}
    grids = []
    edge_values = edges.values
    for i in range(len(edges) - 1):
        # Skip the leading edge in the piece when concatenating
        start = int(i > 0)
        mesh = funcs[scale[i]](dim=edges.dim,
                               start=edge_values[i],
                               stop=edge_values[i + 1],
                               num=num[i] + start,
                               unit=edges.unit)
        grids.append(mesh[edges.dim, start:])

    return sc.concat(grids, edges.dim)
