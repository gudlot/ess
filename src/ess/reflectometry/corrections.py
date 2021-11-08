# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
from scipy.special import erf


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
    scale_factor = erf((sample_size / beam_on_sample * fwhm_to_std).values)
    return sc.array(values=scale_factor, dims=theta.dims)


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
