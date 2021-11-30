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
    https://docs.mantidproject.org/v3.9.0/algorithms/Q1D-v2.html#algm-q1d), but we
    ignore the horizontal `x` component.
    See the schematic in Fig 5 of doi: 10.1016/j.nima.2016.03.007.
    """
    grav = sc.norm(gravity)
    L2 = sc.norm(scattered_beam)
    y = sc.dot(scattered_beam, gravity) / grav
    wavelength = sc.to_unit(wavelength, "m", copy=False)
    drop = grav * m_n**2 / (2 * h**2) * wavelength**2 * L2**2
    return sc.asin(sc.abs(y + drop) / L2) - sc.to_unit(sample_rotation, 'rad')


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
    graph["theta"] = theta
    graph["Q"] = reflectometry_q
    return graph
