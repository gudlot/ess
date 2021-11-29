# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import m_n, h, pi
from scippneutron.tof import conversions
from scippneutron.core.conversions import _elem_dtype


def incident_beam(source_chopper, sample_position):
    return sample_position - source_chopper.value.position


def theta(gravity: sc.Variable, wavelength: sc.Variable, incident_beam: sc.Variable,
          scattered_beam: sc.Variable, sample_rotation: sc.Variable) -> sc.Variable:
    """
    Compute the theta angle, including gravity correction,
    This is similar to the theta calculation in SANS (see
    https://docs.mantidproject.org/v3.9.0/algorithms/Q1D-v2.html#algm-q1d).
    See the schematic in Fig 5 of doi: 10.1016/j.nima.2016.03.007.
    """
    grav = sc.norm(gravity)
    L2 = sc.norm(scattered_beam)
    # In reflectometry, we are only interested in the vertical component of the
    # scattered beam, so we nullify the component which is parallel to the sample
    # orientation.
    # sample_vertical = sc.cross(incident_beam, sample_orientation)
    # sample_vertical /= sc.norm(sample_vertical)
    # sample_to_detector = sc.cross(sample_orientation, sample_vertical)
    # sample_to_detector /= sc.norm(sample_to_detector)
    # scattered_beam_no_horizontal = sc.dot(scattered_beam,
    #                                       sample_to_detector) * sample_to_detector
    y = sc.dot(scattered_beam, gravity) / grav
    # y = sc.dot(scattered_beam_no_horizontal, gravity) / grav
    # n = sc.cross(incident_beam, gravity)
    # n /= sc.norm(n)
    # # x = sc.dot(scattered_beam, n)
    # x = sc.dot(scattered_beam_no_horizontal, n)
    wavelength = sc.to_unit(wavelength, "m", copy=False)
    drop = grav * m_n**2 / (2 * h**2) * wavelength**2 * L2**2
    # omega = sc.norm(sample_orientation)
    return sc.asin(sc.abs(y + drop) / L2) - sc.to_unit(sample_rotation, 'rad')
    # return sc.asin(sc.sqrt(x**2 + (y + drop)**2) / L2)  # - sc.to_unit(omega, 'rad')


# def theta(two_theta: sc.Variable) -> sc.Variable:
#     """
#     Determine the value of theta from the gamma and sample_omega_angle angles.
#     """
#     # return gamma - sc.to_unit(sample_omega_angle, 'rad')
#     return 0.5 * two_theta


def reflectometry_q(wavelength: sc.Variable, theta: sc.Variable) -> sc.Variable:
    """
    Compute the Q vector from the theta angle computed as the difference
    between gamma and omega.
    """
    dtype = _elem_dtype(wavelength)
    c = (4 * pi).astype(dtype)
    return c * sc.sin(theta.astype(dtype, copy=False)) / wavelength


def reflectometry_graph() -> dict:
    """
    Generate a coordinate transformation graph for reflectometry.
    """
    graph = {**conversions.beamline(scatter=True), **conversions.elastic("tof")}
    del graph['two_theta']
    del graph['dspacing']
    del graph['Q']
    del graph['energy']

    # graph["two_theta"] = two_theta
    # # graph["gamma"] = "two_theta"
    graph['incident_beam'] = incident_beam
    graph["theta"] = theta
    graph["Q"] = reflectometry_q
    return graph
