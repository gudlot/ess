# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import m_n, h, g, pi
from scippneutron.tof import conversions
from scippneutron.core.conversions import _elem_dtype


def reflectometry_theta(gravity: sc.Variable, wavelength: sc.Variable,
                        incident_beam: sc.Variable,
                        scattered_beam: sc.Variable) -> sc.Variable:
    """
    Compute the reflectometry theta angle, including gravity correction.
    It is similar to the classical two_theta in other techniques,
    but we also neglect the x component of the scattered beam.
    """
    scattered_beam = scattered_beam.copy()
    scattered_beam.fields.x *= 0
    # Arbitrary internal convention: beam=z, gravity=y
    grav = sc.norm(gravity)
    L2 = sc.norm(scattered_beam)
    y = sc.dot(scattered_beam, gravity) / grav
    n = sc.cross(incident_beam, gravity)
    n /= sc.norm(n)
    x = sc.dot(scattered_beam, n)
    wavelength = sc.to_unit(wavelength, "m", copy=False)
    drop = grav * m_n**2 / (2 * h**2) * wavelength**2 * L2**2
    return sc.asin(sc.sqrt(x**2 + (y + drop)**2) / L2)


def reflectometry_q(wavelength: sc.Variable, two_theta: sc.Variable) -> sc.Variable:
    """
    The reflectometry theta angle corresponds to what is usually called theta in other
    techniques. Hence, the reflectometry q calculation overrides the default by
    taking `sin(two_theta)` instead of `sin(two_theta/2)`.
    """
    dtype = _elem_dtype(wavelength)
    c = (4 * pi).astype(dtype)
    return c * sc.sin(two_theta.astype(dtype, copy=False)) / wavelength


def reflectometry_graph():
    """
    Generate a coordinate transformation graph for reflectometry.
    """
    graph = {**conversions.beamline(scatter=True), **conversions.elastic("tof")}
    graph["two_theta"] = reflectometry_theta
    graph["Q"] = reflectometry_q
    return graph
