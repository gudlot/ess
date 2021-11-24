# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import m_n, h, pi
from scippneutron.tof import conversions
from scippneutron.core.conversions import _elem_dtype


def two_theta(gravity: sc.Variable, wavelength: sc.Variable, incident_beam: sc.Variable,
              scattered_beam: sc.Variable) -> sc.Variable:
    """
    Compute the gamma angle, including gravity correction,
    It is similar to the classical two_theta in other techniques (such as SANS
    https://docs.mantidproject.org/v3.9.0/algorithms/Q1D-v2.html#algm-q1d),
    but we neglect the x component of the scattered beam.
    See the schematic in Fig 5 of doi: 10.1016/j.nima.2016.03.007.
    """
    grav = sc.norm(gravity)
    L2 = sc.norm(scattered_beam)
    y = sc.dot(scattered_beam, gravity) / grav
    n = sc.cross(incident_beam, gravity)
    n /= sc.norm(n)
    wavelength = sc.to_unit(wavelength, "m", copy=False)
    drop = grav * m_n**2 / (2 * h**2) * wavelength**2 * L2**2
    return sc.asin(sc.abs(y + drop) / L2)


# def theta(gamma: sc.Variable, sample_omega_angle: sc.Variable) -> sc.Variable:
#     """
#     Determine the value of theta from the gamma and sample_omega_angle angles.
#     """
#     return gamma - sc.to_unit(sample_omega_angle, 'rad')

# def reflectometry_q(wavelength: sc.Variable, theta: sc.Variable) -> sc.Variable:
#     """
#     Compute the Q vector from the theta angle computed as the difference
#     between gamma and omega.
#     """
#     dtype = _elem_dtype(wavelength)
#     c = (4 * pi).astype(dtype)
#     return c * sc.sin(theta.astype(dtype, copy=False)) / wavelength


def reflectometry_graph() -> dict:
    """
    Generate a coordinate transformation graph for reflectometry.
    """
    graph = {**conversions.beamline(scatter=True), **conversions.elastic("tof")}
    graph["two_theta"] = two_theta
    # graph["gamma"] = "two_theta"
    # graph["theta"] = theta
    # graph["Q"] = reflectometry_q
    return graph
