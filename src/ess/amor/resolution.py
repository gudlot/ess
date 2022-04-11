# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import scipp as sc
from .tools import fwhm_to_std


def wavelength_resolution(chopper_1_position: sc.Variable,
                          chopper_2_position: sc.Variable,
                          pixel_position: sc.Variable) -> sc.Variable:
    """
    Find the wavelength resolution contribution as described in Secion 4.3.3 of the Amor
    publication (doi: 10.1016/j.nima.2016.03.007).

    :param chopper_1_position: Position of first chopper (the one closer to the source).
    :param chopper_2_position: Position of second chopper
        (the one closer to the sample).
    :param pixel_position: Positions for detector pixels.
    :return: The angular resolution variable, as standard deviation.
    """
    distance_between_choppers = (chopper_2_position.data.fields.z -
                                 chopper_1_position.data.fields.z)
    chopper_midpoint = (chopper_1_position.data +
                        chopper_2_position.data) * sc.scalar(0.5)
    chopper_detector_distance = (pixel_position.fields.z - chopper_midpoint.fields.z)
    return fwhm_to_std(distance_between_choppers / chopper_detector_distance)


def sample_size_resolution(pixel_position: sc.Variable,
                           sample_size: sc.Variable) -> sc.Variable:
    """
    The resolution from the projected sample size, where it may be bigger
    than the detector pixel resolution as described in Secion 4.3.3 of the Amor
    publication (doi: 10.1016/j.nima.2016.03.007).

    :param pixel_position: Positions for detector pixels.
    :param sample_size: Size of sample.
    :return: Standard deviation of contribution from the sample size.
    """
    return fwhm_to_std(
        sc.to_unit(sample_size, 'm') / sc.to_unit(pixel_position.fields.z, 'm'))


def angular_resolution(pixel_position: sc.Variable, theta: sc.Variable,
                       detector_spatial_resolution: sc.Variable) -> sc.Variable:
    """
    Determine the angular resolution as described in Secion 4.3.3 of the Amor
    publication (doi: 10.1016/j.nima.2016.03.007).

    :param pixel_position: Positions for detector pixels.
    :param theta: Theta values for events.
    :param detector_spatial_resolution: FWHM of detector pixel resolution.
    """
    return fwhm_to_std(
        sc.to_unit(
            sc.atan(
                sc.to_unit(
                    detector_spatial_resolution,
                    'm') / sc.to_unit(pixel_position.fields.z, 'm')),
            theta.unit)) / theta


def sigma_Q(angular_resolution: sc.Variable, wavelength_resolution: sc.Variable,
            sample_size_resolution: sc.Variable, q_bins: sc.Variable) -> sc.Variable:
    """
    Combine all of the components of the resolution and add Q contribution.

    :param angular_resolution: Angular resolution contribution.
    :param wavelength_resolution: Wavelength resolution contribution.
    :param sample_size_resolution: Sample size resolution contribution.
    :param q_bins: Q-bin values.
    :return: Combined resolution function.
    """
    return sc.sqrt(angular_resolution**2 + wavelength_resolution**2 +
                   sample_size_resolution**2).max('detector_id') * sc.midpoints(q_bins)
