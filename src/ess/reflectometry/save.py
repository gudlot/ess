# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc


def save_ort(data_array: sc.DataArray, filename: str):
    """
    Save a data array with the ORSO .ort file format.

    Parameters
    ----------
    data_array:
        Scipp-data array to save.
    filename:
        Filename.
    """
    from orsopy import fileio
    if filename[:-4] == '.ort':
        raise UserWarning("The expected output file ending is .ort.")
    q = data_array.mean('detector_id').coords['Q']
    if data_array.mean('detector_id').coords.is_edges('Q'):
        q = sc.midpoints(q)
    R = data_array.mean('detector_id').data
    sR = sc.stddevs(data_array.mean('detector_id').data)
    sq = data_array.coords['sigma_Q']
    dataset = fileio.orso.OrsoDataset(
        data_array.attrs['orso'].value,
        np.array([q.values, R.values, sR.values, sq.values]).T)
    fileio.orso.save_orso([dataset], filename)
