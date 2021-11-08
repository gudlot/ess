# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ... import reflectometry as refl


def _tof_correction(data: sc.DaraArray, tau: sc.Variable,
                    chopper_phase: sc.Variable) -> sc.DaraArray:
    """
    A correction for the presense of the chopper with respect to the "true" ToF.
    Also fold the two pulses.
    TODO: generalise mechanism to fold any number of pulses.
    """
    tof_offset = tau * chopper_phase / (180.0 * sc.units.deg)
    # Make 2 bins, one for each pulse
    edges = sc.array(dims=['tof'],
                     values=[
                         -tof_offset.value, (tau - tof_offset).value,
                         (2 * tau - tof_offset).value
                     ],
                     unit=tau.unit)
    data = sc.bin(data, edges=[edges])
    # Make one offset for each bin
    offset = sc.concatenate(tof_offset, tof_offset - tau, 'tof')
    # Apply the offset on both bins
    data.bins.coords['tof'] += offset
    # Rebin to exclude second (empty) pulse range
    return sc.bin(data, edges=[sc.concatenate(0. * sc.units.us, tau, 'tof')])


def load(
    filename,
    sample_angle_offset: sc.Variable = 0 * sc.units.deg,
    beam_size: sc.Variable = 0.001 * sc.units.m,
    sample_size: sc.Variable = 0.01 * sc.units.m,
    detector_spatial_resolution: sc.Variable = 0.0025 * sc.units.m,
    chopper_sample_distance: sc.Variable = 15.0 * sc.units.m,
    chopper_speed: sc.Variable = 20 / 3 * 1e-6 / sc.units.us,
    chopper_detector_distance: sc.Variable = 19.0 * sc.units.m,
    chopper_chopper_distance: sc.Variable = 0.49 * sc.units.m,
    chopper_phase: sc.Variable = -8.0 * sc.units.deg,
    gravity: sc.Variable = sc.vector(value=[0, -1, 0]) * sc.constants.g
) -> sc.DaraArray:
    """
    Loader for a single Amor data file.

    :param sample_angle_offset: Correction for omega or possibly misalignment of sample.
        Default is `0 degrees of arc`.
    :param beam_size: Size of the beam perpendicular to the scattering surface.
        Default is `0.001 m`.
    :param sample_size: Size of the sample in direction of the beam.
        Default :code:`0.01 m`.
    :param detector_spatial_resolution: Spatial resolution of the detector.
        Default is `2.5 mm`.
    :param chopper_sample_distance: Distance from chopper to sample.
        Default `15. m`.
    :param chopper_speed: Rotational velocity of the chopper.
        Default is `6.6666... e-6 µs^{-1}`.
    :param chopper_detector_distance: Distance from chopper to detector.
        Default is `19 m`.
    :param chopper_chopper_distance: The distance between the wavelength defining
        choppers. Default `0.49 m`.
    :param chopper_phase: Phase offset between chopper pulse and ToF zero.
        Default is `-8. degrees of arc`.
    """
    data = refl.io.load(filename=filename,
                        sample_angle_offset=sample_angle_offset,
                        gravity=gravity,
                        beam_size=beam_size,
                        sample_size=sample_size,
                        detector_spatial_resolution=detector_spatial_resolution)

    # Convert tof nanoseconds to microseconds for convenience
    # TODO: is it safe to assume that the dtype of the binned wrapper coordinate is
    # the same as the dtype of the underlying event coordinate?
    if data.coords['tof'].dtype != sc.dtype.float64:
        data.bins.coords['tof'] = data.bins.coords['tof'].astype('float64')
        data.coords['tof'] = data.coords['tof'].astype('float64')
    data.bins.coords['tof'] = sc.to_unit(data.bins.coords['tof'], 'us')
    data.coords['tof'] = sc.to_unit(data.coords['tof'], 'us')

    # These are Amor specific parameters
    tau = 1 / (2 * chopper_speed)
    # The source position is not the true source position due to the
    # use of choppers to define the pulse.
    data.coords["source_position"] = sc.geometry.position(0.0 * sc.units.m,
                                                          0.0 * sc.units.m,
                                                          -chopper_sample_distance)
    data = _tof_correction(data, tau, chopper_phase)
    return data
