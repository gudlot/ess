# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc


class Chopper:
    def __init__(self,
                 frequency,
                 position,
                 phase=None,
                 opening_angles_center=None,
                 opening_angles_width=None,
                 opening_angles_open=None,
                 opening_angles_close=None):
        self.frequency = frequency
        self.position = position
        self.phase = phase
        self.opening_angles_center = opening_angles_center
        self.opening_angles_width = opening_angles_width
        self.opening_angles_open = opening_angles_open
        self.opening_angles_close = opening_angles_close


def _to_angular_frequency(f):
    """
    Convert frequency in Hz to angular frequency.
    """
    return (2.0 * np.pi * sc.units.rad) * f


def _extract_and_concatenate(container, key, dim):
    array = None
    for item in container.values():
        scalar = getattr(item, key)
        if array is None:
            array = scalar
        else:
            array = sc.concatenate(array, scalar, dim)
    return array


def make_chopper_cascade(beamline):
    """
    Create a description of a chopper cascade using a supplied description of
    beamline components.
    """

    for chopper in beamline.choppers.values():
        if chopper.opening_angles_open is None and chopper.opening_angles_close is None:
            chopper.opening_angles_open = (chopper.opening_angles_center -
                                           0.5 * chopper.opening_angles_width)
            chopper.opening_angles_close = (chopper.opening_angles_center +
                                            0.5 * chopper.opening_angles_width)

    ds = sc.Dataset()

    ds["names"] = sc.array(dims=["chopper"], values=list(beamline.choppers.keys()))

    ds["angular_frequency"] = _to_angular_frequency(
        _extract_and_concatenate(container=beamline.choppers,
                                 key="frequency",
                                 dim="chopper"))

    ds["phase"] = sc.to_unit(
        _extract_and_concatenate(container=beamline.choppers,
                                 key="phase",
                                 dim="chopper"), 'rad')

    ds["position"] = _extract_and_concatenate(container=beamline.choppers,
                                              key="position",
                                              dim="chopper")

    for key in ["opening_angles_open", "opening_angles_close"]:
        ds[key] = sc.to_unit(
            _extract_and_concatenate(container=beamline.choppers,
                                     key=key,
                                     dim="chopper"), 'rad')

    return ds
