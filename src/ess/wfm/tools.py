# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import scipp as sc


def to_bin_centers(x, dim):
    """
    Convert array edges to centers
    """
    return 0.5 * (x[dim, 1:] + x[dim, :-1])


def to_bin_edges(x, dim):
    """
    Convert array centers to edges
    """
    idim = x.dims.index(dim)
    if x.shape[idim] < 2:
        one = 1.0 * x.unit
        return sc.concatenate(x[dim, 0:1] - one, x[dim, 0:1] + one, dim)
    else:
        center = to_bin_centers(x, dim)
        # Note: use range of 0:1 to keep dimension dim in the slice to avoid
        # switching round dimension order in concatenate step.
        left = center[dim, 0:1] - (x[dim, 1] - x[dim, 0])
        right = center[dim, -1] + (x[dim, -1] - x[dim, -2])
        return sc.concatenate(sc.concatenate(left, center, dim), right, dim)


def _angular_frame_edge_to_time(angular_frequency, angle, phase):
    """
    Convert an angle on a rotating chopper to a time point (in microseconds).
    """
    div = angular_frequency * (1.0 * sc.units.s)
    return (angle + phase) / div * (1.0e6 * sc.units.us)


def get_frame_properties(frame):
    """
    Get coordinates of a chopper frame opening in time and distance.
    """
    pos = sc.norm(frame["position"])
    tstart = _angular_frame_edge_to_time(frame["angular_frequency"],
                                         frame["opening_angles_open"],
                                         frame["phase"])
    tend = _angular_frame_edge_to_time(frame["angular_frequency"],
                                       frame["opening_angles_close"],
                                       frame["phase"])
    return pos, tstart, tend
