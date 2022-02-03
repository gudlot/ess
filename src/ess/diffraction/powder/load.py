# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipp as sc
import scippneutron as scn

from .corrections import subtract_empty_instrument


def _load_aux_file_as_wavelength(filename):
    da = scn.load(filename,
                  advanced_geometry=True,
                  load_pulse_times=False,
                  mantid_args={'LoadMonitors': True})
    return da.transform_coords('wavelength',
                               graph={
                                   **scn.tof.conversions.beamline(scatter=True),
                                   **scn.tof.conversions.elastic("tof")
                               })


def load_vanadium(vanadium_file, empty_instrument_file):
    # TODO normalize by proton charge?
    vanadium = _load_aux_file_as_wavelength(vanadium_file)
    empty_instrument = _load_aux_file_as_wavelength(empty_instrument_file)
    return subtract_empty_instrument(vanadium, empty_instrument)


def _as_boolean_mask(var):
    if var.dtype in ('float32', 'float64'):
        if sc.any(var != var.to(dtype='int64')).value:
            raise ValueError(
                'Cannot construct boolean mask, the input mask has fractional values.')
    return var.to(dtype=bool)


def load_calibration(filename: Union[str, Path],
                     *,
                     mantid_args: Optional[dict] = None) -> sc.Dataset:
    """
    Function that loads calibration files using the Mantid algorithm
    LoadDiffCal. This algorithm produces up to three workspaces, a
    TableWorkspace containing conversion factors between TOF and d, a
    GroupingWorkspace with detector groups and a MaskWorkspace for masking.
    The information from the TableWorkspace and GroupingWorkspace is converted
    to a Scipp dataset and returned, while the MaskWorkspace is ignored for
    now. Only the keyword parameters Filename and InstrumentName are mandatory.

    Note that this function requires mantid to be installed and available in
    the same Python environment as ess.

    :param filename: The name of the calibration file to be loaded.
    :param mantid_args : Dictionary with arguments for the
                         LoadDiffCal Mantid algorithm.
                         Currently, InstrumentName or InstrumentFilename
                         is required.
    :raises: If the InstrumentName given in mantid_args is not
             valid.
    :return: A Dataset containing the calibration data and grouping.
    """

    mantid_args = {} if mantid_args is None else mantid_args

    with scn.mantid.run_mantid_alg('LoadDiffCal', Filename=str(filename),
                                   **mantid_args) as ws:
        ds = scn.from_mantid(ws.OutputCalWorkspace)

        # Note that despite masking and grouping stored in separate workspaces,
        # there is no need to handle potentially mismatching ordering: All
        # workspaces have been created by the same algorithm, which should
        # guarantee ordering.
        mask_ws = ws.OutputMaskWorkspace
        rows = mask_ws.getNumberHistograms()
        mask = sc.array(dims=['row'],
                        values=np.fromiter((mask_ws.readY(i)[0] for i in range(rows)),
                                           count=rows,
                                           dtype=np.bool_))
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
    # TODO conversion to int32 required to match type in detector_info
    #      loaded by Mantid. Do we ultimately need that compatibility?
    #      If int32 large enough for everything we do as ESS?
    ds.coords['detector'] = ds['detid'].data.astype('int32')
    del ds['detid']

    return ds
