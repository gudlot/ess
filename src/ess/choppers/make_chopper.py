# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
from . import utils
import numpy as np


def make_chopper(frequency: sc.Variable,
                 position: sc.Variable,
                 phase: sc.Variable = None,
                 cutout_angles_center: sc.Variable = None,
                 cutout_angles_width: sc.Variable = None,
                 cutout_angles_begin: sc.Variable = None,
                 cutout_angles_end: sc.Variable = None,
                 kind: str = None) -> sc.Dataset:

    data = {"frequency": frequency, "position": position}
    if phase is not None:
        data["phase"] = phase
    if cutout_angles_center is not None:
        data["cutout_angles_center"] = cutout_angles_center
    if cutout_angles_width is not None:
        data["cutout_angles_width"] = cutout_angles_width
    if cutout_angles_begin is not None:
        data["cutout_angles_begin"] = cutout_angles_begin
    if cutout_angles_end is not None:
        data["cutout_angles_end"] = cutout_angles_end
    if kind is not None:
        data["kind"] = kind
    chopper = sc.Dataset(data=data)

    # Sanitize input parameters
    widths = utils.cutout_angles_width(chopper)
    if (sc.min(widths) < sc.scalar(0.0, unit=widths.unit)).value:
        raise ValueError("Negative window width found in chopper cutout angles.")
    if not np.all(np.diff(utils.cutout_angles_begin(chopper).values) > 0):
        raise ValueError("Chopper begin cutout angles are not monotonic.")
    if not np.all(np.diff(utils.cutout_angles_end(chopper).values) > 0):
        raise ValueError("Chopper end cutout angles are not monotonic.")

    return chopper
