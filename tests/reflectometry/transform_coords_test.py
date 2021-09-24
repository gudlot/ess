import scipp as sc
from scipp.constants import g, h, neutron_mass
from ess.reflectometry import transform_coords
import numpy as np


def test_y_dash_with_different_secondary_flight_paths():
    sample_position = sc.vector(value=[0, 0, 0], unit=sc.units.m)
    detector_position = sc.vector(value=[0, 0.5, 1], unit=sc.units.m)

    # Approximate cold-neutron velocities
    vel = 1000 * (sc.units.m / sc.units.s)
    wav = sc.to_unit(h / (vel * neutron_mass),
                     unit=sc.units.angstrom)

    # In this setup the faster the neutrons the closer d'y(z) tends to 1.0
    grad = transform_coords.to_y_dash(wavelength=wav, sample_position=sample_position,
                                      detector_position=detector_position)

    scattered_beam = detector_position - sample_position
    no_gravity_grad = scattered_beam.fields.y / scattered_beam.fields.z
    gravity_effect_grad = (-0.5 * g * scattered_beam.fields.z / (vel * vel))
    assert sc.isclose(grad, no_gravity_grad + gravity_effect_grad).value


def test_y_dash_with_different_velocities():
    sample_position = sc.vector(value=[0, 0, 0], unit=sc.units.m)
    detector_position = sc.vector(value=[0, 1, 1], unit=sc.units.m)

    vel = 1000 * (sc.units.m / sc.units.s)
    wav = sc.to_unit(h / (vel * neutron_mass),
                     unit=sc.units.angstrom)

    # In this setup the faster the neutrons the closer d'y(z) tends to 1.0
    grad = transform_coords.to_y_dash(wavelength=wav,
                                      sample_position=sample_position,
                                      detector_position=detector_position)
    assert sc.less(grad, 1 * sc.units.one).value

    vel *= 2
    wav = sc.to_unit(h / (vel * neutron_mass),
                     unit=sc.units.angstrom)
    grad_fast = transform_coords.to_y_dash(wavelength=wav,
                                           sample_position=sample_position,
                                           detector_position=detector_position)
    # Testing that gravity has greater influence on slow neutrons.
    assert sc.less(grad, grad_fast).value


def _angle(a, b):
    return sc.acos(sc.dot(a, b) / (sc.norm(a) * sc.norm(b)))


def test_scattering_angle():
    sample_position = sc.vector(value=[0, 0, 0], unit=sc.units.m)
    detector_position = sc.vector(value=[0, 1, 1], unit=sc.units.m)
    scattered_beam = detector_position - sample_position
    beam_direction = sc.vector(value=[0, 0, 1], unit=sc.units.m)
    no_gravity_angle = _angle(scattered_beam, beam_direction)

    vel = 1000 * (sc.units.m / sc.units.s)
    wav = sc.to_unit(h / (vel * neutron_mass),
                     unit=sc.units.angstrom)

    angle = transform_coords.to_scattering_angle(w=0 * sc.units.rad,
                                                 wavelength=wav,
                                                 detector_id=None,
                                                 position=detector_position,
                                                 sample_position=sample_position)
    assert sc.less(angle, no_gravity_angle).value

    gravity_shift_y = -0.5 * g * (scattered_beam.fields.z ** 2 / vel ** 2)
    expected = _angle(scattered_beam + gravity_shift_y
                      * sc.vector(value=[0, 1, 0]), beam_direction)
    assert sc.isclose(angle, expected).value


def test_det_wavelength_to_wavelength_scattering_angle():
    # comparible with cold-neutrons from moderator
    vel = 2000 * (sc.units.m / sc.units.s)
    wav = sc.to_unit(h / (vel * neutron_mass),
                     unit=sc.units.angstrom)
    sample_position = sc.vector(value=[0, 0, 0], unit=sc.units.m)
    source_position = sc.vector(value=[0, 0, -10], unit=sc.units.m)
    detector_position = sc.vector(value=[0, 1, 1], unit=sc.units.m)

    coords = {}
    coords["sample_position"] = sample_position
    coords["source_position"] = source_position
    coords["position"] = detector_position
    coords["wavelength"] = wav
    coords["w"] = 0.0 * sc.units.rad
    coords["detector_id"] = 0.0 * sc.units.one
    measurement = sc.DataArray(data=1.0 * sc.units.one, coords=coords)

    settings = {'scattering_angle': transform_coords.to_scattering_angle}
    transformed = sc.transform_coords(x=measurement,
                                      coords=['wavelength', 'scattering_angle'],
                                      graph=settings)
    assert sc.isclose(transformed.coords['scattering_angle'],
                      (np.pi / 4) * sc.units.rad, atol=1e-4 * sc.units.rad).value

    # We now check the sample angle. Setting to previous final scattering angle
    # should yield a scattering angle of 0.
    measurement.coords["w"] = transformed.coords['scattering_angle']
    transformed = sc.transform_coords(x=measurement,
                                      coords=['wavelength', 'scattering_angle'],
                                      graph=settings)
    assert sc.isclose(transformed.coords['scattering_angle'],
                      0.0 * sc.units.rad).value
