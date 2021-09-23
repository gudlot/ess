import scipp as sc
from scipp import constants

def to_velocity(wavelength):
    return sc.to_unit(constants.h / (wavelength * constants.neutron_mass ), sc.units.m / sc.units.s,
                   copy=False)

# Derivative of y with respect to z
def to_y_dash(wavelength, sample_position, detector_position):
    z_origin = sample_position.fields.z
    y_origin = sample_position.fields.y
    z_measured = detector_position.fields.z
    y_measured = detector_position.fields.y
    z_diff = z_measured - z_origin
    y_diff = y_measured - y_origin
    velocity_sq = to_velocity(wavelength)
    velocity_sq *= velocity_sq
    g = sc.vector(value=[0, -1, 0]) * sc.constants.g
    # dy due to gravity = -0.5gt^2 = -0.5g(dz/dv)^2
    # therefore y'(z) = dy/dz - 0.5g.dz/dv^2 / dz
    return (-0.5 * sc.norm(g) * z_diff / velocity_sq) + (y_diff / z_diff)

# Question, we consume the detector_id on the basis we no longer need it, but should we do this?
def to_scattering_angle(w, wavelength, detector_id, position, sample_position):
    z_origin = sample_position.fields.z
    y_origin = sample_position.fields.y
    z_measured = position.fields.y
    y_dash = to_y_dash(wavelength, sample_position, position)
    height =  y_dash * (z_measured - z_origin) + y_origin
    return -w + sc.atan2(x=height, y=z_measured)

#def test_det_wavelength_to_wavelength_scattering_angle():
#    # comparible with cold-neutrons from moderator
#    vel = 2000 * (sc.units.m /sc.units.s)
#    wav = sc.to_unit(constants.h / (vel * constants.neutron_mass))
#    sample_position = sc.vector(value=[0, 0, 0], unit=sc.units.m)
#    source_position = sc.vector(value=[0, 0, -10], unit=sc.units.m)
#    detector_position = sc.vector(value=[0, 1, 1], unit=sc.units.m)
#    L1 = sc.norm(sample_position - source_position)
#    L2 = sc.norm(detector_position - sample_position)
#    LTotal = L1 + L2
#    tof = LTotal / vel
#
#    measurement = sc.Dataset()
#    measurement.coords["sample_position"] = sc.vector(value=[0, 0, 0], unit=sc.units.m)
#    measurement.coords["source_position"] = sc.vector(value=[0, 0, -10], unit=sc.units.m)
#    measurement.coords["position"] = sc.vector(value=[0, 0, 1], unit=sc.units.m)
#    measurement.coords["wavelength"] = wav
#
#    data = # Need beamline with instrument
#    transformed = gravity.to_scattering_wavelength()

def test_y_dash_with_different_secondary_flight_paths():
    sample_position = sc.vector(value=[0, 0, 0], unit=sc.units.m)
    detector_position = sc.vector(value=[0, 0.5, 1], unit=sc.units.m)
    L2 = sc.norm(sample_position - detector_position)

    # Approximate cold-neutron velocities
    vel = 1000 * (sc.units.m /sc.units.s)
    wav = sc.to_unit(constants.h / (vel * constants.neutron_mass), unit=sc.units.angstrom)

    # In this setup the faster the neutrons the closer d'y(z) tends to 1.0
    grad = to_y_dash(wavelength=wav, sample_position=sample_position, detector_position=detector_position)

    scattered_beam = detector_position - sample_position
    no_gravity_grad = scattered_beam.fields.y / scattered_beam.fields.z
    gravity_effect = (-0.5 * constants.g * scattered_beam.fields.z / (vel * vel))
    print()
    print(grad.value)
    print((no_gravity_grad + gravity_effect).value)
    assert sc.isclose(grad, no_gravity_grad + gravity_effect).value

def test_y_dash_with_different_velocities():
    sample_position = sc.vector(value=[0, 0, 0], unit=sc.units.m)
    detector_position = sc.vector(value=[0, 1, 1], unit=sc.units.m)

    # Approximate cold-neutron velocities
    vel = 1000 * (sc.units.m /sc.units.s)
    wav = sc.to_unit(constants.h / (vel * constants.neutron_mass), unit=sc.units.angstrom)

    # In this setup the faster the neutrons the closer d'y(z) tends to 1.0
    grad = to_y_dash(wavelength=wav, sample_position=sample_position, detector_position=detector_position)
    assert sc.less(grad, 1 * sc.units.one).value

    vel *= 2
    wav = sc.to_unit(constants.h / (vel * constants.neutron_mass), unit=sc.units.angstrom)
    grad_fast = to_y_dash(wavelength=wav, sample_position=sample_position, detector_position=detector_position)
    # Testing that gravity has greater influence on slow neutrons.
    assert sc.less(grad, grad_fast).value

