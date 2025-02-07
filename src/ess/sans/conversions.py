# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

import scipp as sc
from scipp.constants import m_n, h
from scippneutron.tof import conversions
from scippneutron.core.conversions import _elem_unit


def two_theta(gravity: sc.Variable, wavelength: sc.Variable, incident_beam: sc.Variable,
              scattered_beam: sc.Variable) -> sc.Variable:
    grav = sc.norm(gravity)
    L2 = sc.norm(scattered_beam)
    y = sc.dot(scattered_beam, gravity) / grav
    n = sc.cross(incident_beam, gravity)
    n /= sc.norm(n)
    x_term = sc.dot(scattered_beam, n)
    x_term *= x_term

    y_term = sc.to_unit(wavelength, _elem_unit(L2), copy=True)
    y_term *= y_term
    drop = L2**2
    drop *= grav * (m_n**2 / (2 * h**2))
    # Optimization when handling either the dense or the event coord of binned data:
    # - For the event coord, both operands have same dims, and we can multiply in place
    # - For the dense coord, we need to broadcast using non in-place operation
    if set(drop.dims).issubset(set(y_term.dims)):
        y_term *= drop
    else:
        y_term = drop * y_term
    y_term += y
    y_term *= y_term

    if set(x_term.dims).issubset(set(y_term.dims)):
        y_term += x_term
    else:
        y_term = y_term + x_term
    out = sc.sqrt(y_term, out=y_term)
    out /= L2
    out = sc.asin(out, out=out)
    return out


def sans_elastic(gravity: bool = False) -> dict:
    """
    Generate a coordinate transformation graph for SANS elastic scattering.
    By default, the effects of gravity on the neutron flight paths are not included.

    :param gravity: Take into account the bending of the neutron flight paths from the
        Earth's gravitational field if ``True``.
    """
    graph = {**conversions.beamline(scatter=True), **conversions.elastic_Q("tof")}
    if gravity:
        graph["two_theta"] = two_theta
    return graph


def sans_monitor() -> dict:
    """
    Generate a coordinate transformation graph for SANS monitor (no scattering).
    """
    return {
        **conversions.beamline(scatter=False),
        **conversions.elastic_wavelength("tof")
    }
