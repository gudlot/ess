# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import pi


def cutout_angles_begin(chopper: sc.Dataset, unit="rad") -> sc.Variable:
    if "cutout_angles_begin" in chopper:
        out = chopper["cutout_angles_begin"].data
    elif all(x in chopper for x in ["cutout_angles_width", "cutout_angles_center"]):
        out = chopper[
            "cutout_angles_center"].data - 0.5 * chopper["cutout_angles_width"].data
    else:
        raise KeyError("Chopper does not contain the information required to compute "
                       "the cutout_angles_begin.")
    return sc.to_unit(out, unit, copy=False)


def cutout_angles_end(chopper: sc.Dataset, unit="rad") -> sc.Variable:
    if "cutout_angles_end" in chopper:
        out = chopper["cutout_angles_end"].data
    elif all(x in chopper for x in ["cutout_angles_width", "cutout_angles_center"]):
        out = chopper[
            "cutout_angles_center"].data + 0.5 * chopper["cutout_angles_width"].data
    else:
        raise KeyError("Chopper does not contain the information required to compute "
                       "the cutout_angles_end.")
    return sc.to_unit(out, unit, copy=False)


def cutout_angles_width(chopper: sc.Dataset, unit="rad") -> sc.Variable:
    if "cutout_angles_width" in chopper:
        out = chopper["cutout_angles_width"].data
    elif all(x in chopper for x in ["cutout_angles_begin", "cutout_angles_end"]):
        out = chopper["cutout_angles_end"].data - chopper["cutout_angles_begin"].data
    else:
        raise KeyError("Chopper does not contain the information required to compute "
                       "the cutout_angles_width.")
    return sc.to_unit(out, unit, copy=False)


def cutout_angles_center(chopper: sc.Dataset, unit="rad") -> sc.Variable:
    if "cutout_angles_center" in chopper:
        out = chopper["cutout_angles_center"].data
    elif all(x in chopper for x in ["cutout_angles_begin", "cutout_angles_end"]):
        out = 0.5 * (chopper["cutout_angles_begin"].data +
                     chopper["cutout_angles_end"].data)
    else:
        raise KeyError("Chopper does not contain the information required to compute "
                       "the cutout_angles_center.")
    return sc.to_unit(out, unit, copy=False)


def angular_frequency(chopper):
    return (2.0 * sc.units.rad) * pi * chopper["frequency"].data


def time_open(chopper, unit="us"):
    return sc.to_unit(
        (cutout_angles_begin(chopper) + sc.to_unit(chopper["phase"].data, "rad")) /
        angular_frequency(chopper),
        unit,
        copy=False)


def time_closed(chopper, unit="us"):
    return sc.to_unit(
        (cutout_angles_end(chopper) + sc.to_unit(chopper["phase"].data, "rad")) /
        angular_frequency(chopper),
        unit,
        copy=False)


def find_chopper_keys(da):
    return [key for key in da.coords if key.lower().startswith("chopper")]
