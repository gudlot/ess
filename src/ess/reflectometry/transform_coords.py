import scipp as sc
from scipp.constants import neutron_mass, h, g


def to_velocity(wavelength):
    return sc.to_unit(h / (wavelength * neutron_mass),
                      sc.units.m / sc.units.s,
                      copy=False)


# Derivative of y with respect to z
def to_y_dash(wavelength, sample_position, detector_position):
    diff = (detector_position - sample_position)
    velocity_sq = to_velocity(wavelength)
    velocity_sq *= velocity_sq
    gy = sc.vector(value=[0, -1, 0]) * g
    # dy due to gravity = -0.5gt^2 = -0.5g(dz/dv)^2
    # therefore y'(z) = dy/dz - 0.5g.dz/dv^2 / dz
    return (-0.5 * sc.norm(gy) * diff.fields.z / velocity_sq) + (diff.fields.y /
                                                                 diff.fields.z)


def to_scattering_angle(w, wavelength, detector_id, position, sample_position):
    z_origin = sample_position.fields.z
    y_origin = sample_position.fields.y
    z_measured = position.fields.z
    y_dash = to_y_dash(wavelength, sample_position, position)
    height = y_dash * (z_measured - z_origin) + y_origin
    return sc.atan2(y=height, x=z_measured) - w
