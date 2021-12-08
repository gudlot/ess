# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import m_n, h, pi
from scippneutron.tof import conversions
from scippneutron.core.conversions import _elem_dtype


def theta(gravity: sc.Variable, wavelength: sc.Variable, incident_beam: sc.Variable,
          scattered_beam: sc.Variable, sample_rotation: sc.Variable) -> sc.Variable:
    """
    Compute the theta angle, including gravity correction,
    This is similar to the theta calculation in SANS (see
    https://docs.mantidproject.org/nightly/algorithms/Q1D-v2.html#q-unit-conversion),
    but we ignore the horizontal `x` component.
    See the schematic in Fig 5 of doi: 10.1016/j.nima.2016.03.007.
    """
    grav = sc.norm(gravity)
    L2 = sc.norm(scattered_beam)
    y = sc.dot(scattered_beam, gravity) / grav
    wavelength = sc.to_unit(wavelength, "m", copy=False)
    drop = wavelength**2 * (L2**2)
    # Note: make use of in-place operations for better performance
    drop *= grav * (m_n**2 / (2 * h**2))
    drop += y
    return sc.asin(sc.abs(drop) / L2) - sc.to_unit(sample_rotation, 'rad')


def reflectometry_q(wavelength: sc.Variable, theta: sc.Variable) -> sc.Variable:
    """
    Compute the Q vector from the theta angle computed as the difference
    between gamma and omega.
    Note that this is different from the 'normal' Q defined in scippneutron where
    `two_theta / 2` is used, while we use `theta` (computed above) directly here.
    """
    dtype = _elem_dtype(wavelength)
    c = (4 * pi).astype(dtype)
    return c * sc.sin(theta.astype(dtype, copy=False)) / wavelength


def specular_reflection() -> dict:
    """
    Generate a coordinate transformation graph for specular reflection reflectometry.
    """
    graph = {**conversions.beamline(scatter=True), **conversions.elastic("tof")}
    del graph['two_theta']
    del graph['dspacing']
    del graph['Q']
    del graph['energy']
    graph["theta"] = theta
    graph["Q"] = reflectometry_q
    return graph
