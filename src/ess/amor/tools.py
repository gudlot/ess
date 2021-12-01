# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
from typing import Union


def q_grid(dim: str,
           edges: Union[list, np.ndarray],
           scale: Union[list, str],
           num: Union[list, int],
           unit: str = None) -> sc.Variable:
    """
    Generate bin edges for a `q_grid` inspired by the custom linear-log grid from
    Jochen Stahn at PSI.

    :param dim: The dimension of the ouptut Variable.
    :param edges: The edges for the different parts of the mesh.
    :param scale: A string or list of strings specifying the scaling for the different
        parts of the mesh. Possible values for the scaling are `"linear"` and `"log"`.
        If a list is supplied, the length of the list must be one less than the length
        of the `edges` parameter.
    :param num: An integer or a list of integers specifying the number of points to use
        in each part of the mesh. If a list is supplied, the length of the list must be
        one less than the length of the `edges` parameter.
    :param unit: The unit of the ouptut Variable.
    """
    if not isinstance(scale, list):
        scale = [scale]
    if not isinstance(num, list):
        num = [num]
    if len(scale) != len(edges) - 1:
        raise ValueError("Sizes do not match")

    funcs = {"linear": sc.linspace, "log": sc.geomspace}
    grids = []
    for i in range(len(edges) - 1):
        # Skip the leading edge in the piece when concatenating
        start = int(i > 0)
        mesh = funcs[scale[i]](dim=dim,
                               start=edges[i],
                               stop=edges[i + 1],
                               num=num[i] + start,
                               unit=unit)
        grids.append(mesh[dim, start:])

    return sc.concat(grids, dim)
