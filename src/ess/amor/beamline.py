# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import g


def make_beamline(
    sample_rotation: sc.Variable = None,
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

    :param sample_rotation: Sample rotation (omega) angle.
        Default is `None`.
    :type sample_rotation: Variable.
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

    :returns: A dict.
    :rtype: dict
    """
    beamline = {
        'sample_rotation': sample_rotation,
        'beam_size': beam_size,
        'sample_size': sample_size,
        'detector_spatial_resolution': detector_spatial_resolution,
        'gravity': gravity
    }
    # TODO: in scn.load_nexus, the chopper parameters are stored as coordinates
    # of a DataArray, and the data value is a string containing the name of the
    # chopper. This does not allow storing e.g. chopper cutout angles.
    # We should change this to be a Dataset, which is what we do here.
    beamline["source_chopper"] = sc.scalar(
        sc.Dataset(
            data={
                'frequency': chopper_frequency,
                'phase': chopper_phase,
                'position': chopper_position
            }))
    return beamline


def instrument_view_components(da: sc.DataArray):
    return {
        "sample": {
            'center': da.meta['sample_position'],
            'color': 'red',
            'size': sc.vector(value=[0.2, 0.01, 0.2], unit=sc.units.m),
            'type': 'box'
        },
        "source_chopper": {
            'center': da.meta['source_chopper'].value["position"].data,
            'color': 'blue',
            'size': sc.vector(value=[0.5, 0, 0], unit=sc.units.m),
            'type': 'disk'
        }
    }
