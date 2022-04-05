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
    q = sc.midpoints(data_array.mean('detector_id').coords['Q']).values
    R = data_array.mean('detector_id').data.values
    sR = sc.stddevs(data_array.mean('detector_id').data).values
    sq = data_array.coords['sigma_Q_by_Q'].max('detector_id').values * q
    dataset = fileio.orso.OrsoDataset(data_array.attrs['orso'].value,
                                      np.array([q, R, sR, sq]).T)
    fileio.orso.save_orso([dataset], filename)
