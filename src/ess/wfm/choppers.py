# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc


class Beamline:
    def __init__(self, choppers, source):
        self.choppers = choppers
        self.source = source


class Chopper:
    def __init__(self,
                 frequency,
                 distance,
                 phase=None,
                 frame_center=None,
                 frame_width=None,
                 frame_start=None,
                 frame_end=None):
        self.frequency = frequency
        self.distance = distance
        self.phase = phase
        self.frame_center = frame_center
        self.frame_width = frame_width
        self.frame_start = frame_start
        self.frame_end = frame_end


# def _deg_to_rad(x):
#     """
#     Convert degrees to radians.
#     """
#     return x * (np.pi * sc.units.rad / (180.0 * sc.units.deg))


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


def choppers(beamline):
    """
    Create a description of a chopper cascade using a supplied description of
    beamline components.

    To override the default chopper parameters, you can supply a dict
    containing the chopper name ("WFM1" or "FOL2") and a sub-dict of parameters
    such as frequency or tdc.
    For example:
        choppers({"WFM1": {"frequency": 17., "phase": 55.}})
    Frequencies are in Hz and angles are in degrees.
    For components distances, we assume that the origin is the source double
    chopper, and the direction of the beam is along `z`.
    """

    # inventory = {key: [] for key in list(default_choppers.values())[0]}
    # inventory = {}
    for chopper in beamline.choppers.values():
        if chopper.frame_start is None and chopper.frame_end is None:
            chopper.frame_start = chopper.frame_center - 0.5 * chopper.frame_width
            chopper.frame_end = chopper.frame_center + 0.5 * chopper.frame_width
        # for key, value in chopper.items():
        #     if key not in inventory:
        #         inventory[key] = []
        #     inventory[key].append(value)

    ds = sc.Dataset()

    ds["names"] = sc.array(dims=["chopper"],
                           values=list(beamline.choppers.keys()))

    ds["angular_frequency"] = _to_angular_frequency(
        _extract_and_concatenate(container=beamline.choppers,
                                 key="frequency",
                                 dim="chopper"))

    ds["phase"] = sc.to_unit(
        _extract_and_concatenate(container=beamline.choppers,
                                 key="phase",
                                 dim="chopper"), 'rad')

    ds["distance"] = _extract_and_concatenate(container=beamline.choppers,
                                              key="distance",
                                              dim="chopper")

    # tdc_array = np.array(inventory["tdc"]).reshape(
    #     ds["choppers"].sizes["chopper"], 1)

    for key in ["frame_start", "frame_end"]:
        ds[key] = sc.to_unit(
            _extract_and_concatenate(container=beamline.choppers,
                                     key=key,
                                     dim="chopper"), 'rad')

    # ds["pulse_length"] = beamline.source["pulse_length"]
    # ds["pulse_t_0"] = beamline.source["pulse_t_0"]
    # ds["source_position"] = beamline.source["distance"]

    return ds, beamline.source
