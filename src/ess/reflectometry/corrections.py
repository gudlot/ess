# flake8: noqa: E501
"""
Corrections to be used for neutron reflectometry reduction processes.
"""
import numpy as np
import scipp as sc
from scipy.special import erf
from ess.reflectometry import HDM, G_ACC


def angle_with_gravity(data, pixel_position, sample_position):
    """
    Find the angle of reflection when accounting for the presence of gravity.

    Args:
        data (`scipp._scipp.core.DataArray`): Reduction data array.
        pixel_position (`scipp._scipp.core.VariableView`): Detector pixel positions, should be a `vector_3_float64`-type object.
        sample_position (`scipp._scipp.core.VariableView`): Scattered neutron origin position.

    Returns:
        (`scipp._scipp.core.Variable`): Gravity corrected angle values.
    """
    # This is a workaround until scipp #1819 is resolved, at which time the following should be used instead
    # velocity = sc.to_unit(HDM / wavelength, 'm/s')
    # At which point the args can be changed to wavelength
    # (where this is data.bins.constituents['data'].coords['wavelength'].astype(sc.dtype.float64) or similar)
    # instead of data
    velocity = sc.to_unit(
        HDM / data.bins.constituents["data"].coords["wavelength"].astype(
            sc.dtype.float64),
        "m/s",
    )
    data.bins.constituents["data"].coords["velocity"] = velocity
    velocity = data.bins.coords["velocity"]
    velocity.unit = sc.units.m / sc.units.s
    y_measured = sc.geometry.y(pixel_position)
    z_measured = sc.geometry.z(pixel_position)
    z_origin = sc.geometry.z(sample_position)
    y_origin = sc.geometry.y(sample_position)
    y_dash = y_dash0(velocity, z_origin, y_origin, z_measured, y_measured)
    intercept = y_origin - y_dash * z_origin
    y_true = z_measured * y_dash + intercept
    angle = sc.to_unit(
        sc.atan(y_true / z_measured).bins.constituents["data"], 'deg')
    return angle


def y_dash0(velocity, z_origin, y_origin, z_measured, y_measured):
    """
    Evaluation of the first dervative of the kinematic equations for for the trajectory of a neutron reflected from a surface.

    Args:
        velocity (`scipp._scipp.core.VariableView`): Neutron velocity.
        z_origin (`scipp._scipp.core.Variable`): The z-origin position for the reflected neutron.
        y_origin (`scipp._scipp.core.Variable`): The y-origin position for the reflected neutron.
        z_origin (`scipp._scipp.core.Variable`): The z-measured position for the reflected neutron.
        y_origin (`scipp._scipp.core.Variable`): The y-measured position for the reflected neutron.

    Returns:
        (`scipp._scipp.core.VariableView` or `array_like`): The gradient of the trajectory of the neutron at the origin position.
    """
    velocity2 = velocity * velocity
    z_diff = z_measured - z_origin
    y_diff = y_measured - y_origin
    return -0.5 * sc.norm(G_ACC) * z_diff / velocity2 + y_diff / z_diff


def illumination_correction(beam_size, sample_size, theta):
    """
    The factor by which the intensity should be multiplied to account for the
    scattering geometry, where the beam is Gaussian in shape.

    Args:
        beam_size (:py:attr:`sc.Variable`): Width of incident beam.
        sample_size (:py:attr:`sc.Variable`): Width of sample in the dimension of the beam.
        theta (:py:attr:`sc.Variable`): Incident angle.

    Returns:
        (:py:attr:`array_like`): Correction factor.
    """
    beam_on_sample = beam_size / sc.sin(theta)
    fwhm_to_std = 2 * np.sqrt(2 * np.log(2))
    scale_factor = erf((sample_size / beam_on_sample * fwhm_to_std).values)
    return sc.Variable(values=scale_factor, dims=theta.dims)


def illumination_of_sample(beam_size, sample_size, theta):
    """
    Determine the illumination of the sample by the beam and therefore the size of this illuminated length.

    Args:
        beam_size (:py:attr:`sc.Variable`): Width of incident beam, in metres.
        sample_size (:py:attr:`sc.Variable`): Width of sample in the dimension of the beam, in metres.
        theta (:py:attr:`sc.Variable`): Incident angle.

    Returns:
        (`sc.Variable`): The size of the beam, for each theta, on the sample.
    """
    beam_on_sample = beam_size / sc.sin(theta)
    if ((sc.mean(beam_on_sample)) > sample_size).value:
        beam_on_sample = sc.broadcast(sample_size,
                                      shape=theta.shape,
                                      dims=theta.dims)
        # beam_on_sample = sc.ones(shape=theta.shape,
        #                          unit=sc.units.dimensionless,
        #                          dims=theta.dims) * sample_size
    return beam_on_sample
