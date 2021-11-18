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

        # # Sanitize input parameters
        # if (sc.min(self.opening_angles_width) < sc.scalar(
        #         0.0, unit=self.opening_angles_width.unit)).value:
        #     raise ValueError("Negative window width found in chopper opening angles.")
        # lengths = []
        # for angles in [
        #         opening_angles_center, opening_angles_width, opening_angles_open,
        #         opening_angles_close
        # ]:
        #     if angles is not None:
        #         lengths.append(len(angles))
        # if lengths.count(lengths[0]) != len(lengths):
        #     raise ValueError("All angle input arrays (centers, widths, open or close) "
        #                      "must have the same length.")
        # if not np.all(np.diff(self.opening_angles_open.values) > 0):
        #     raise ValueError("Chopper opening angles are not monotonic.")
        # if not np.all(np.diff(self.opening_angles_close.values) > 0):
        #     raise ValueError("Chopper closing angles are not monotonic.")

    def __eq__(self, other):
        """
        Define == operator to allow for coordinate comparison.
        This will also be called for the != operator.
        """
        return all([
            sc.identical(self.frequency, other.frequency),
            sc.identical(self.position, other.position),
            sc.identical(self.phase, other.phase),
            sc.identical(self.opening_angles_center, other.opening_angles_center),
            sc.identical(self.opening_angles_width, other.opening_angles_width),
            sc.identical(self.opening_angles_open, other.opening_angles_open),
            sc.identical(self.opening_angles_close, other.opening_angles_close),
            self.kind == other.kind
        ])

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
    def phase(self):
        return sc.to_unit(self._phase, sc.units.rad)

    @phase.setter
    def phase(self, value):
        self._phase = value

    @property
    def opening_angles_center(self):
        if self._opening_angles_center is None:
            out = 0.5 * (self._opening_angles_open + self._opening_angles_close)
        else:
            out = self._opening_angles_center
        return sc.to_unit(out, sc.units.rad)

    @opening_angles_center.setter
    def opening_angles_center(self, value):
        self._opening_angles_center = value

    @property
    def opening_angles_width(self):
        if self._opening_angles_width is None:
            out = self._opening_angles_close - self._opening_angles_open
        else:
            out = self._opening_angles_width
        return sc.to_unit(out, sc.units.rad)

    @opening_angles_width.setter
    def opening_angles_width(self, value):
        self._opening_angles_width = value

    @property
    def opening_angles_open(self):
        if self._opening_angles_open is None:
            out = self._opening_angles_center - 0.5 * self._opening_angles_width
        else:
            out = self._opening_angles_open
        return sc.to_unit(out, sc.units.rad)

    @opening_angles_open.setter
    def opening_angles_open(self, value):
        self._opening_angles_open = value

    @property
    def opening_angles_close(self):
        if self._opening_angles_close is None:
            out = self._opening_angles_center + 0.5 * self._opening_angles_width
        else:
            out = self._opening_angles_close
        return sc.to_unit(out, sc.units.rad)

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
    def time_open(self):
        return sc.to_unit(
            (self.opening_angles_open + self.phase) / self.angular_frequency,
            sc.units.us)

    @property
    def time_close(self):
        return sc.to_unit(
            (self.opening_angles_close + self.phase) / self.angular_frequency,
            sc.units.us)
