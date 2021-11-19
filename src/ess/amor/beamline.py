# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import g
from ..choppers import Chopper


def make_beamline(
    sample_omega_angle: sc.Variable = 0 * sc.units.deg,
    beam_size: sc.Variable = 0.001 * sc.units.m,
    sample_size: sc.Variable = 0.01 * sc.units.m,
    detector_spatial_resolution: sc.Variable = 0.0025 * sc.units.m,
    gravity: sc.Variable = sc.vector(value=[0, -1, 0]) * g,
    chopper_frequency: sc.Variable = sc.scalar(20 / 3, unit='Hz'),
    chopper_phase: sc.Variable = sc.scalar(-8.0, unit='deg'),
    chopper_position: sc.Variable = sc.vector(value=[0, 0, -15.0], unit='m')
) -> dict:
    """
    Amor beamline components.

    :param sample_omega_angle: Sample tilt (omega) angle.
        Default is `0 degrees of arc`.
    :param beam_size: Size of the beam perpendicular to the scattering surface.
        Default is `0.001 m`.
    :param sample_size: Size of the sample in direction of the beam.
        Default :code:`0.01 m`.
    :param detector_spatial_resolution: Spatial resolution of the detector.
        Default is `2.5 mm`.
    :param gravity: Vector representing the direction and magnitude of the Earth's
        gravitational field. Default is `[0, -g, 0]`.
    :param chopper_frequency: Rotational frequency of the chopper.
        Default is `6.6666... Hz`.
    :param chopper_phase: Phase offset between chopper pulse and ToF zero.
        Default is `-8. degrees of arc`.
    :param chopper_position: Position of the chopper.
        Default is `-15 m`.
    """
    beamline = {
        'sample_omega_angle': sample_omega_angle,
        'beam_size': beam_size,
        'sample_size': sample_size,
        'detector_spatial_resolution': detector_spatial_resolution,
        'gravity': gravity
    }
    beamline["source_chopper"] = sc.scalar(
        Chopper(frequency=chopper_frequency,
                phase=chopper_phase,
                position=chopper_position))
    # The source position is not the true source position due to the
    # use of choppers to define the pulse.
    beamline["source_position"] = beamline["source_chopper"].value.position
    return beamline
