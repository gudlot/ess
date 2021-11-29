# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
import scippneutron as scn
from .beamline import make_beamline
import warnings


def _tof_correction(data: sc.DataArray, dim: str = 'tof') -> sc.DataArray:
    """
    A correction for the presense of the chopper with respect to the "true" ToF.
    Also fold the two pulses.
    TODO: generalise mechanism to fold any number of pulses.
    """
    tau = sc.to_unit(1 / (2 * data.coords['source_chopper'].value.frequency),
                     data.coords[dim].unit)
    chopper_phase = data.coords['source_chopper'].value.phase
    tof_offset = tau * chopper_phase / (180.0 * sc.units.deg)
    # Make 2 bins, one for each pulse
    edges = sc.concat([-tof_offset, tau - tof_offset, 2 * tau - tof_offset], dim)
    data = sc.bin(data, edges=[sc.to_unit(edges, data.coords[dim].unit)])
    # Make one offset for each bin
    offset = sc.concat([tof_offset, tof_offset - tau], dim)
    # Apply the offset on both bins
    data.bins.coords[dim] += offset
    # Rebin to exclude second (empty) pulse range
    return sc.bin(data, edges=[sc.concat([0. * sc.units.us, tau], dim)])


def load(filename,
         beamline: dict = make_beamline(),
         disable_warnings: bool = True) -> sc.DataArray:
    """
    Loader for a single Amor data file.

    :param beamline: A dict defining the beamline parameters.
    :param disable_warnings: Do not show warnings from file loading if `True`.
        Default is `True`.
    """
    if disable_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            data = scn.load_nexus(filename)
    else:
        data = scn.load_nexus(filename)

    # Convert tof nanoseconds to microseconds for convenience
    # TODO: is it safe to assume that the dtype of the binned wrapper coordinate is
    # the same as the dtype of the underlying event coordinate?
    data.bins.coords['tof'] = data.bins.coords['tof'].astype('float64', copy=False)
    data.coords['tof'] = data.coords['tof'].astype('float64', copy=False)
    data.bins.coords['tof'] = sc.to_unit(data.bins.coords['tof'], 'us', copy=False)
    data.coords['tof'] = sc.to_unit(data.coords['tof'], 'us', copy=False)

    # Add beamline parameters
    for key, value in beamline.items():
        data.coords[key] = value

    # Perform tof correction and fold two pulses
    return _tof_correction(data)
