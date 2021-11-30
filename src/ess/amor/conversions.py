# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ..reflectometry.conversions import reflectometry_graph


def incident_beam(source_chopper: sc.Variable,
                  sample_position: sc.Variable) -> sc.Variable:
    """
    Compute the incident beam vector from the source chopper position vector,
    instead of the source_position vector.
    """
    return sample_position - source_chopper.value.position


def specular_reflection_graph() -> dict:
    """
    Generate a coordinate transformation graph for Amor reflectometry.
    """
    graph = reflectometry_graph()
    graph['incident_beam'] = incident_beam
    return graph
