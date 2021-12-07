# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import pi


def _cutout_angles(chopper, suffix):
    key = "cutout_angles_" + suffix
    if key in chopper:
        return chopper[key]
    if all(x in chopper for x in ["cutout_angles_width", "cutout_angles_center"]):
        return chopper["cutout_angles_center"].data + (
            int(suffix == "close") - 0.5) * chopper["cutout_angles_width"].data
    raise KeyError("Chopper does not contain the information required to compute "
                   "the cutout angles.")


def cutout_angles_open(chopper: sc.Dataset, unit="rad") -> sc.Variable:
    return sc.to_unit(_cutout_angles(chopper=chopper, suffix="open"), unit)


def cutout_angles_close(chopper: sc.Dataset, unit="rad") -> sc.Variable:
    return sc.to_unit(_cutout_angles(chopper=chopper, suffix="close"), unit)


def angular_frequency(chopper):
    return (2.0 * sc.units.rad) * pi * chopper["frequency"].data


def time_open(chopper, unit="us"):
    return sc.to_unit(
        (cutout_angles_open(chopper) + sc.to_unit(chopper["phase"].data, "rad")) /
        angular_frequency(chopper), unit)


def time_closed(chopper, unit="us"):
    return sc.to_unit(
        (cutout_angles_close(chopper) + sc.to_unit(chopper["phase"].data, "rad")) /
        angular_frequency(chopper), unit)


def find_chopper_keys(da):
    return [key for key in da.coords if key.lower().startswith("chopper")]
