# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import g
from ..choppers import make_chopper
from ..logging import log_call


@log_call(instrument='amor',
          message='Constructing AMOR beamline from default parameters')
def make_beamline(
    sample_rotation: sc.Variable = 45.0 * sc.units.deg,
    beam_size: sc.Variable = 2.0 * sc.units.mm,
    sample_size: sc.Variable = 10.0 * sc.units.mm,
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
    beam_on_sample = beam_size / sc.sin(sample_rotation)
    fwhm_to_std = sc.scalar(2.) * sc.sqrt(sc.scalar(2.) * sc.log(sc.scalar(2.)))
    footprint_scale = sc.erf(sample_size / beam_on_sample * fwhm_to_std)
    beamline = {
        'sample_rotation': sample_rotation,
        'beam_size': beam_size,
        'sample_size': sample_size,
        'detector_spatial_resolution': detector_spatial_resolution,
        'gravity': gravity,
        'footprint_scale': footprint_scale
    }
    # TODO: in scn.load_nexus, the chopper parameters are stored as coordinates
    # of a DataArray, and the data value is a string containing the name of the
    # chopper. This does not allow storing e.g. chopper cutout angles.
    # We should change this to be a Dataset, which is what we do here.
    beamline["source_chopper"] = sc.scalar(
        make_chopper(frequency=chopper_frequency,
                     phase=chopper_phase,
                     position=chopper_position))
    return beamline


@log_call(instrument='amor', level='DEBUG')
def instrument_view_components(da: sc.DataArray) -> dict:
    """
    Create a dict of instrument view components, containing:
      - the sample
      - the source chopper

    :param da: The DataArray containing the sample and source chopper coordinates.
    """
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
