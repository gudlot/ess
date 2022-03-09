# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
"""
File loading for diffraction data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import scipp as sc
import scippneutron as scn

from ..logging import get_logger


@dataclass
class _AuxData:
    da: sc.DataArray
    name: str


def _load_aux_file(filename: Union[str, Path], *, data_name: str) -> _AuxData:
    get_logger('diffraction').info('Loading %s from file %s.', data_name, filename)
    return _AuxData(da=scn.load(filename,
                                advanced_geometry=False,
                                load_pulse_times=False,
                                mantid_args={'LoadMonitors': True}),
                    name=data_name)


def _normalize_by_proton_charge_in_place(data: _AuxData, charge_name: str):
    charge = data.da.meta[charge_name]
    get_logger('diffraction').info('Normalizing %s by proton charge %e%s', data.name,
                                   charge.value, charge.unit)
    data.da /= charge


def _common_edges(*edges, dim: str) -> sc.Variable:
    """
    The data has separate bin edges for each spectrum:

        ^      |---|
    spectrum  |--|
        v        |--|
              < dim >

    This function computes common edges for all spectra
    for the combination of all inputs:

        ^     | --- |
    spectrum  |--   |
        v     |   --|
              < dim >
    """
    lo = sc.reduce([e[dim, 0] for e in edges]).min().min()
    hi = sc.reduce([e[dim, -1] for e in edges]).max().max()
    return sc.concat([lo, hi], dim)


def _replace_by_common_edges(data: List[_AuxData], dim: str):
    for d in data:
        if d.da.coords[dim].sizes != {dim: 2}:
            raise RuntimeError(
                f"Cannot process vanadium data, coordinate '{dim}' of dataset "
                f"'{d.name}' must have sizes {{{dim}: 2}}, "
                f"got {d.da.coords[dim].sizes}")
    edges = _common_edges(*(d.da.coords[dim] for d in data), dim=dim)
    for d in data:
        d.da.coords[dim] = edges


def _filter_bad_pulses(data: _AuxData, gd_proton_charge_name,
                       proton_charge_name) -> _AuxData:
    charge = data.da.meta[proton_charge_name].value.data
    limit = data.da.meta[gd_proton_charge_name].to(unit=charge.unit)
    if (charge.max() < limit).value:
        get_logger('diffraction').info(
            "Discarding data for '%s' because its proton charge "
            'is too low. Maximum charge: %e%s vs good charge: %e%s', data.name,
            charge.max().value, charge.unit, limit.value, limit.unit)
        out = data.da.copy(deep=True)
        for key in ('begin', 'end'):
            out.bins.constituents[key][...] = sc.zeros_like(out.bins.constituents[key])
        return _AuxData(da=out, name=data.name)
    return data


def load_and_preprocess_vanadium(vanadium_file: Union[str, Path],
                                 empty_instrument_file: Union[str, Path],
                                 gd_proton_charge_name: str,
                                 proton_charge_name: str) -> sc.DataArray:
    """
    Load and return data from a vanadium measurement.

    Subtracts events recorded for the instrument without sample.

    Parameters
    ----------
    vanadium_file:
        File that contains the vanadium data.
    empty_instrument_file:
        File that contains data for the empty instrument.
        Must correspond to the same setup as `vanadium_file`.
    gd_proton_charge_name:
        Name of the metadata item in both vanadium and empty instrument
        that stores the average proton charge.
    proton_charge_name:
        Name of the metadata item in the empty instrument
        that stores the time-dependent proton charge.

    Returns
    -------
    :
        (Vanadium - empty instrument) with a wavelength coordinate.
    """
    data = [
        _load_aux_file(vanadium_file, data_name='vanadium'),
        _load_aux_file(empty_instrument_file, data_name='empty instrument')
    ]
    # TODO this also filters out vanadium!
    data = [
        _filter_bad_pulses(d, gd_proton_charge_name, proton_charge_name) for d in data
    ]
    _replace_by_common_edges(data, dim='tof')
    tof_to_wavelength = {
        **scn.tof.conversions.beamline(scatter=True),
        **scn.tof.conversions.elastic("tof")
    }
    for d in data:
        _normalize_by_proton_charge_in_place(d, gd_proton_charge_name)
        d.da = d.da.transform_coords('wavelength', graph=tof_to_wavelength)

    return data[0].da.bins.concat(data[1].da)
