# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
from ..amor.tools import fwhm_to_std


def illumination_correction(beam_size: sc.Variable, sample_size: sc.Variable,
                            theta: sc.Variable) -> sc.Variable:
    """
    Compute the factor by which the intensity should be multiplied to account for the
    scattering geometry, where the beam is Gaussian in shape.

    :param beam_size: Width of incident beam.
    :param sample_size: Width of sample in the dimension of the beam.
    :param theta: Incident angle.
    """
    beam_on_sample = beam_size / sc.sin(theta)
    fwhm_to_std = 2 * np.sqrt(2 * np.log(2))
    return sc.erf(sample_size / beam_on_sample * fwhm_to_std)


def illumination_of_sample(beam_size: sc.Variable, sample_size: sc.Variable,
                           theta: sc.Variable) -> sc.Variable:
    """
    Determine the illumination of the sample by the beam and therefore the size of its
    illuminated length.

    :param beam_size: Width of incident beam.
    :param sample_size: Width of sample in the dimension of the beam.
    :param theta: Incident angle.
    """
    beam_on_sample = beam_size / sc.sin(theta)
    if ((sc.mean(beam_on_sample)) > sample_size).value:
        beam_on_sample = sc.broadcast(sample_size, shape=theta.shape, dims=theta.dims)
    return beam_on_sample


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
        data_array_fp_correction.attrs[
            'orso'].value.reduction.corrections += ['footprint correction']
    except KeyError:
        raise UserWarning("To store information about corrections "
                          "it is necessary to install the orsopy package.")
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
        raise UserWarning("For metadata to be logged in the data array, "
                          "it is necessary to install the orsopy package.")
    return norm
