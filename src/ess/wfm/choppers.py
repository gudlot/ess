# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
from enum import Enum


class ChopperKind(Enum):
    WFM = 1
    FRAME_OVERLAP = 2
    BAND_PASS = 3


class Chopper:
    def __init__(self,
                 frequency: sc.Variable,
                 position: sc.Variable,
                 phase: sc.Variable = None,
                 opening_angles_center: sc.Variable = None,
                 opening_angles_width: sc.Variable = None,
                 opening_angles_open: sc.Variable = None,
                 opening_angles_close: sc.Variable = None,
                 kind: ChopperKind = None):
        self._frequency = frequency
        self._position = position
        self._phase = phase
        self._opening_angles_center = opening_angles_center
        self._opening_angles_width = opening_angles_width
        self._opening_angles_open = opening_angles_open
        self._opening_angles_close = opening_angles_close
        self._kind = kind

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._frequency = value

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def phase(self, as_unit='rad'):
        return sc.to_unit(self._phase, as_unit)

    @phase.setter
    def phase(self, value):
        self._phase = value

    @property
    def opening_angles_center(self, as_unit='rad'):
        if self._opening_angles_center is None:
            out = sc.mean(
                sc.concatenate(self._opening_angles_open, self._opening_angles_close,
                               'none'))
        else:
            out = self._opening_angles_center
        return sc.to_unit(out, as_unit)

    @opening_angles_center.setter
    def opening_angles_center(self, value):
        self._opening_angles_center = value

    @property
    def opening_angles_width(self, as_unit='rad'):
        if self._opening_angles_width is None:
            out = self._opening_angles_close - self._opening_angles_open
        else:
            out = self._opening_angles_width
        return sc.to_unit(out, as_unit)

    @opening_angles_width.setter
    def opening_angles_width(self, value):
        self._opening_angles_width = value

    @property
    def opening_angles_open(self, as_unit='rad'):
        if self._opening_angles_open is None:
            out = self._opening_angles_center - 0.5 * self._opening_angles_width
        else:
            out = self._opening_angles_open
        return sc.to_unit(out, as_unit)

    @opening_angles_open.setter
    def opening_angles_open(self, value):
        self._opening_angles_open = value

    @property
    def opening_angles_close(self, as_unit='rad'):
        if self._opening_angles_close is None:
            out = self._opening_angles_center + 0.5 * self._opening_angles_width
        else:
            out = self._opening_angles_close
        return sc.to_unit(out, as_unit)

    @opening_angles_close.setter
    def opening_angles_close(self, value):
        self._opening_angles_close = value

    @property
    def kind(self):
        return self._kind

    @property
    def angular_frequency(self):
        return (2.0 * np.pi * sc.units.rad) * self._frequency

    @property
    def time_open(self, as_unit='us'):
        return sc.to_unit(
            (self.opening_angles_open + self.phase) / self.angular_frequency, as_unit)

    @property
    def time_close(self, as_unit='us'):
        return sc.to_unit(
            (self.opening_angles_close + self.phase) / self.angular_frequency, as_unit)

        # return (2.0 * np.pi * sc.units.rad) * self._frequency


# def _to_angular_frequency(f: sc.Variable) -> sc.Variable:
#     """
#     Convert frequency in Hz to angular frequency.
#     """
#     return (2.0 * np.pi * sc.units.rad) * f

# def _extract_and_concatenate(container: dict, key: str, dim: str) -> sc.Variable:
#     array = None
#     for item in container.values():
#         scalar = getattr(item, key)
#         if array is None:
#             array = scalar
#         else:
#             array = sc.concatenate(array, scalar, dim)
#     return array

# def make_chopper_cascade(choppers: dict) -> sc.Dataset:
#     """
#     Create a description of a chopper cascade using a supplied description of
#     beamline components.
#     """

#     for chopper in choppers.values():
#         if chopper.opening_angles_open is None and chopper.opening_angles_close is None:
#             chopper.opening_angles_open = (chopper.opening_angles_center -
#                                            0.5 * chopper.opening_angles_width)
#             chopper.opening_angles_close = (chopper.opening_angles_center +
#                                             0.5 * chopper.opening_angles_width)

#     ds = sc.Dataset()

#     ds["names"] = sc.array(dims=["chopper"], values=list(choppers.keys()))

#     ds["angular_frequency"] = _to_angular_frequency(
#         _extract_and_concatenate(container=choppers, key="frequency", dim="chopper"))

#     ds["phase"] = sc.to_unit(
#         _extract_and_concatenate(container=choppers, key="phase", dim="chopper"), 'rad')

#     ds["position"] = _extract_and_concatenate(container=choppers,
#                                               key="position",
#                                               dim="chopper")

#     for key in ["opening_angles_open", "opening_angles_close"]:
#         ds[key] = sc.to_unit(
#             _extract_and_concatenate(container=choppers, key=key, dim="chopper"), 'rad')

#     return ds
