# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
"""
File loading for diffraction data.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import scipp as sc
import scippneutron as scn

from ..logging import get_logger
from .corrections import normalize_by_proton_charge, subtract_empty_instrument


def _load_aux_file_as_wavelength(filename: Union[str, Path]) -> sc.DataArray:
    da = scn.load(filename,
                  advanced_geometry=True,
                  load_pulse_times=False,
                  mantid_args={'LoadMonitors': True})
    return da.transform_coords('wavelength',
                               graph={
                                   **scn.tof.conversions.beamline(scatter=True),
                                   **scn.tof.conversions.elastic("tof")
                               })


def load_and_preprocess_vanadium(
        vanadium_file: Union[str, Path],
        empty_instrument_file: Union[str, Path]) -> sc.DataArray:
    """
    Load and return data from a vanadium measurement.

    Subtracts events recorded for the instrument without sample.

    :param vanadium_file: File that contains the vanadium data.
    :param empty_instrument_file: File that contains data for the empty instrument.
                                  Must correspond to the same setup as `vanadium_file`.
    :return: Vanadium - empty instrument with a wavelength coordinate.
    """
    get_logger('diffraction').info(
        'Loading vanadium from file %s\n'
        'and correcting by empty instrument from file %s', vanadium_file,
        empty_instrument_file)
    vanadium = _load_aux_file_as_wavelength(vanadium_file)
    empty_instrument = _load_aux_file_as_wavelength(empty_instrument_file)
    normalize_by_proton_charge(vanadium, in_place=True)
    normalize_by_proton_charge(empty_instrument, in_place=True)
    return subtract_empty_instrument(vanadium, empty_instrument)


def _as_boolean_mask(var: sc.Variable) -> sc.Variable:
    if var.dtype in ('float32', 'float64'):
        if sc.any(var != var.to(dtype='int64')).value:
            raise ValueError(
                'Cannot construct boolean mask, the input mask has fractional values.')
    return var.to(dtype=bool)


def _parse_calibration_instrument_args(
    filename: Union[str, Path],
    *,
    instrument_filename: Optional[str] = None,
    instrument_name: Optional[str] = None,
) -> Dict[str, str]:
    if instrument_filename is not None:
        if instrument_name is not None:
            raise ValueError('Only one argument of `instrument_name` and '
                             '`instrument_filename` is allowed, got both.')
        instrument_arg = {'InstrumentFilename': instrument_filename}
        instrument_message = f'with instrument file {instrument_filename}'
    else:
        if instrument_name is None:
            raise ValueError('Need one argument of `instrument_name` and '
                             '`instrument_filename` is allowed, got neither.')
        instrument_arg = {'InstrumentName': instrument_name}
        instrument_message = f'with instrument {instrument_name}'

    get_logger('diffraction').info('Loading calibration from file %s %s', filename,
                                   instrument_message)
    return instrument_arg


def load_calibration(filename: Union[str, Path],
                     *,
                     instrument_filename: Optional[str] = None,
                     instrument_name: Optional[str] = None,
                     mantid_args: Optional[dict] = None) -> sc.Dataset:
    """
    Load and return calibration data.

    Uses the Mantid algorithm `LoadDiffCal
    <https://docs.mantidproject.org/nightly/algorithms/LoadDiffCal-v1.html>`_
    and stores the data in a :class:`scipp.Dataset`

    Note that this function requires mantid to be installed and available in
    the same Python environment as ess.

    :param filename: The name of the calibration file to load.
    :param instrument_filename: Instrument definition file.
    :param instrument_name: Name of the instrument.
    :param mantid_args: Dictionary with additional arguments for the
                        `LoadDiffCal` Mantid algorithm.
    :return: A Dataset containing the calibration data and masking.
    """

    mantid_args = {} if mantid_args is None else mantid_args
    mantid_args.update(
        _parse_calibration_instrument_args(filename,
                                           instrument_filename=instrument_filename,
                                           instrument_name=instrument_name))

    with scn.mantid.run_mantid_alg('LoadDiffCal', Filename=str(filename),
                                   **mantid_args) as ws:
        ds = scn.from_mantid(ws.OutputCalWorkspace)
        mask_ws = ws.OutputMaskWorkspace
        rows = mask_ws.getNumberHistograms()
        mask = sc.array(dims=['row'],
                        values=np.fromiter((mask_ws.readY(i)[0] for i in range(rows)),
                                           count=rows,
                                           dtype=np.bool_),
                        unit=None)
    # This is deliberately not stored as a mask since that would make
    # subsequent handling, e.g., with groupby, more complicated. The mask
    # is conceptually not masking rows in this table, i.e., it is not
    # marking invalid rows, but rather describes masking for other data.
    ds["mask"] = _as_boolean_mask(mask)

    # The file does not define units
    # TODO why those units? Can we verify?
    ds["difc"].unit = 'us / angstrom'
    ds["difa"].unit = 'us / angstrom**2'
    ds["tzero"].unit = 'us'

    ds = ds.rename_dims({'row': 'detector'})
    ds.coords['detector'] = ds['detid'].data
    del ds['detid']

    return ds
