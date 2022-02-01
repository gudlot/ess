# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipp as sc
import scippneutron as scn


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
        group_ws = ws.OutputGroupingWorkspace
        rows = mask_ws.getNumberHistograms()
        mask = sc.array(dims=['row'],
                        values=np.fromiter((mask_ws.readY(i)[0] for i in range(rows)),
                                           count=rows,
                                           dtype=np.bool_))
        group = sc.array(dims=['row'],
                         values=np.fromiter((group_ws.readY(i)[0] for i in range(rows)),
                                            count=rows,
                                            dtype=np.int32))
    # This is deliberately not stored as a mask since that would make
    # subsequent handling, e.g., with groupby, more complicated. The mask
    # is conceptually not masking rows in this table, i.e., it is not
    # marking invalid rows, but rather describes masking for other data.
    ds["mask"] = _as_boolean_mask(mask)
    ds["group"] = group

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


def find_spectra(*, calibration, detector_info):
    # Merge cal with detector-info, which contains information on how
    # `dataset` groups its detectors. At the same time, the coord
    # comparison in `merge` ensures that detector IDs of `dataset` match
    # those of `calibration`.
    cal = sc.merge(detector_info, calibration)

    # Masking and grouping information in the calibration table interferes
    # with `groupby.mean`, dropping.
    # TODO still true? Seems to work
    # for name in ("mask", "group"):
    #     if name in cal:
    #         del cal[name]

    # Translate detector-based calibration information into coordinates
    # of data. We are hard-coding some information here: the existence of
    # "spectra", since we require labels named "spectrum" and a
    # corresponding dimension. Given that this is in a branch that is
    # access only if "detector_info" is present this should probably be ok.
    cal = sc.groupby(cal, group='spectrum').mean('detector')
    # `mean` turns the mask into floats, convert back.
    cal['mask'] = _as_boolean_mask(cal['mask'].data)
    return cal


def merge_calibration(*, into: sc.DataArray, calibration: sc.Dataset) -> sc.DataArray:
    dim = calibration.dim
    if not sc.identical(into.coords[dim], calibration.coords[dim]):
        raise ValueError(
            f'Coordinate {dim} of calibration and target dataset do not agree.')
    out = into.copy(deep=False)
    for name in ('difa', 'difc', 'tzero', 'group'):
        out.attrs[name] = calibration[name].data
    out.masks['cal'] = calibration['mask'].data
    return out


def calibrate_and_focus(*, sample, calibration):
    # Import here to avoid cycle from importing at module level.
    from .conversions import to_dspacing_with_calibration

    sample = merge_calibration(into=sample, calibration=calibration)
    dspacing = to_dspacing_with_calibration(sample)
    return sc.groupby(dspacing, group='group').bins.concat('spectrum')
