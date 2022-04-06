# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import m_n, h, pi
from scippneutron.tof import conversions
from scippneutron.core.conversions import _elem_dtype, _elem_unit


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
    y_correction = sc.to_unit(wavelength, _elem_unit(L2), copy=True)
    y_correction *= y_correction
    drop = L2**2
    drop *= grav * (m_n**2 / (2 * h**2))
    # Optimization when handling either the dense or the event coord of binned data:
    # - For the event coord, both operands have same dims, and we can multiply in place
    # - For the dense coord, we need to broadcast using non in-place operation
    if set(drop.dims).issubset(set(y_correction.dims)):
        y_correction *= drop
    else:
        y_correction = y_correction * drop
    y_correction += y
    out = sc.abs(y_correction, out=y_correction)
    out /= L2
    out = sc.asin(out, out=out)
    out -= sc.to_unit(sample_rotation, 'rad')
    return out


def reflectometry_q(wavelength: sc.Variable, theta: sc.Variable) -> sc.Variable:
    """
    Compute the Q vector from the theta angle computed as the difference
    between gamma and omega.
    Note that this is identical the 'normal' Q defined in scippneutron, except that
    the `theta` angle is given as an input instead of `two_theta`.
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


def tof_to_wavelength(
    data_array: sc.DataArray,
    wavelength_edges: sc.Variable,
    graph: dict = specular_reflection()) -> sc.DataArray:
    """
    Use transform coords to convert from ToF to wavelength, cutoff high and
    low limits for wavelength, and add necessary ORSO metadata.

    :param data_array: Data array to convert.
    :param wavelength_edges: The lower and upper limits for the wavelength.
    :return: New data array with wavelength dimension.
    """
    data_array_wav = data_array.transform_coords(["wavelength"], graph=graph)
    data_array_wav = sc.bin(data_array_wav, edges=[wavelength_edges])
    try:
        from orsopy import fileio
        data_array_wav.attrs[
            'orso'].value.data_source.measurement.instrument_settings.wavelength = fileio.base.ValueRange(
                wavelength_edges.min().value,
                wavelength_edges.max().value, 'angstrom')
    except ImportError:
        raise UserWarning("For metadata to be logged in the data array, "
                          "it is necessary to install the orsopy package.")
    return data_array_wav
