# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
"""
File loading for diffraction data.
"""

from pathlib import Path
from typing import Union

import scipp as sc
import scippneutron as scn

from ..logging import get_logger
from .corrections import subtract_empty_instrument


def _load_aux_file_as_wavelength(filename: Union[str, Path]) -> sc.DataArray:
    da = scn.load(filename,
                  advanced_geometry=False,
                  load_pulse_times=False,
                  mantid_args={'LoadMonitors': True})
    return da.transform_coords('wavelength',
                               graph={
                                   **scn.tof.conversions.beamline(scatter=True),
                                   **scn.tof.conversions.elastic("tof")
                               })


def _remove_based_on_proton_charge(da: sc.DataArray, gd_proton_charge_name,
                                   proton_charge_name) -> bool:
    charge = da.meta[proton_charge_name].value.data
    limit = da.meta[gd_proton_charge_name].to(unit=charge.unit)
    if (charge.max() < limit).value:
        get_logger('diffraction').info(
            'Discarding data for the empty instrument because its proton charge '
            'is too low. Vanadium will not be corrected for background. '
            'Maximum charge: %e%s vs good charge: %e%s',
            charge.max().value, charge.unit, limit.value, limit.unit)
        return True
    return False


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
    get_logger('diffraction').info('Loading vanadium from file %s.', vanadium_file)
    vanadium = _load_aux_file_as_wavelength(vanadium_file)
    vanadium /= vanadium.meta[gd_proton_charge_name]
    empty_instrument = _load_aux_file_as_wavelength(empty_instrument_file)
    if _remove_based_on_proton_charge(empty_instrument, gd_proton_charge_name,
                                      proton_charge_name):
        # TODO This is a hack because in the POWGEN test data, all events are
        #  filtered out for the empty instrument. Ultimately, we need to properly
        #  filter pulses by proton charge when that functionality is available.
        return vanadium
    get_logger('diffraction').info(
        'Subtracting empty instrument loaded from file %s from vanadium.',
        empty_instrument_file)
    empty_instrument /= empty_instrument.meta[gd_proton_charge_name]
    return subtract_empty_instrument(vanadium, empty_instrument)
