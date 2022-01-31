# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipp as sc
import scippneutron as scn


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
    ds["mask"] = mask
    ds["group"] = group

    # The file does not define units
    # TODO why those units? Can we verify?
    ds["difc"].unit = 'us / angstrom'
    ds["difa"].unit = 'us / angstrom**2'
    ds["tzero"].unit = 'us'

    ds = ds.rename_dims({'row': 'detector'})
    ds.coords['detector'] = ds['detid'].data
    del ds['detid']

    return ds


def merge_calibration(*, into: sc.DataArray, calibration: sc.Dataset) -> sc.DataArray:
    res = into.copy(deep=False)
    # TODO compare detector / spectrum
    for name in ('difa', 'difc', 'tzero'):
        res.attrs[name] = calibration[name].data
    return res
