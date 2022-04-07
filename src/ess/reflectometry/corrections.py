# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import warnings
import numpy as np
import scipp as sc
from ..amor.tools import fwhm_to_std


def footprint_correction(data_array: sc.DataArray) -> sc.DataArray:
    """
    Perform the footprint correction on the data array that has a :code:`beam_size` and
    binned :code:`theta` values.

    :param data_array: Data array to perform footprint correction on.
    :return: Footprint corrected data array.
    """
    beam_on_sample = data_array.coords['beam_size'] / sc.sin(
        data_array.bins.coords['theta'])
    footprint_scale = sc.erf(
        fwhm_to_std(data_array.coords['sample_size'] / beam_on_sample))
    data_array_fp_correction = data_array / footprint_scale.squeeze()
    try:
        data_array_fp_correction.attrs['orso'].value.reduction.corrections += [
            'footprint correction'
        ]
    except KeyError:
        warnings.warn("To store information about corrections it is "
                      "necessary to install the orsopy package.", UserWarning)
    return data_array_fp_correction


def normalise_by_counts(data_array: sc.DataArray) -> sc.DataArray:
    """
    Normalise the bin-summed data by the total number of counts.

    :param data_array: Data array to be normalised.
    :return: Normalised data array.
    """
    ncounts = data_array.sum()
    norm = data_array / ncounts
    try:
        norm.attrs['orso'].value.reduction.corrections += ['total counts']
    except KeyError:
        warnings.warn("To store information about corrections it is "
                      "necessary to install the orsopy package.", UserWarning)
    return norm
