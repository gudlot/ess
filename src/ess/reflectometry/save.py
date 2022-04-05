# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc
from orsopy import fileio


def save(data_array: sc.DataArray, filename: str):
    """
    Save a data array with the ORSO .ort file format.

    :param data_array: Scipp-data array to save.
    :param filename: Filename.
    """
    if filename[:-4] == '.ort':
        raise UserWarning("The expected output file ending is .ort.")
    q = sc.midpoints(data_array.mean('detector_id').coords['Q'])
    R = data_array.mean('detector_id').data
    sR = sc.stddevs(data_array.mean('detector_id').data)
    sq = data_array.coords['sigma_Q_by_Q'].max('detector_id') * q
    dataset = fileio.orso.OrsoDataset(data_array.attrs['orso'].value,
                                      np.array([q.values, R.values,
                                                sR.values, sq.values]).T)
    fileio.orso.save_orso([dataset], filename)
