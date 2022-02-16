# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
"""
Coordinate transformations for diffraction.
"""

from typing import Optional

import scipp as sc

from .corrections import merge_calibration


def _dspacing_from_diff_calibration_generic_impl(t, t0, a, c):
    """
    This function implements the solution to
      t = a * d^2 + c * d + t0
    for a != 0.
    It uses the following way of expressing the solution with an order of operations
    that is optimized for low memory usage.
      d = (sqrt([x-t0+t] / x) - 1) * c / (2a)
      x = c^2 / (4a)
    """
    x = c**2 / (4 * a)
    out = (x - t0) + t
    out /= x
    del x
    sc.sqrt(out, out=out)
    out -= 1
    out *= c / (2 * a)
    return out


def dspacing_from_diff_calibration(tof: sc.Variable, tzero: sc.Variable,
                                   difa: sc.Variable, difc: sc.Variable) -> sc.Variable:
    r"""
    Compute d-spacing from calibration parameters.

    d-spacing is the positive solution of

    .. math:: \mathsf{tof} = \mathsf{DIFA} * d^2 + \mathsf{DIFC} * d + t_0

    This function can be used with :func:`scipp.transform_coords`.

    :seealso: :func:`ess.diffraction.conversions.to_dspacing_with_calibration`
    """
    if sc.all(difa == sc.scalar(0.0, unit=difa.unit)).value:
        return (tof - tzero) / difc
    return _dspacing_from_diff_calibration_generic_impl(tof, tzero, difa, difc)


def _restore_tof_if_in_wavelength(data: sc.DataArray) -> sc.DataArray:
    if 'wavelength' not in data.dims:
        return data

    # TODO better error message
    # TODO remove empty graph
    tof_data = data.transform_coords('tof', graph={'_': '_'})
    del tof_data.coords['wavelength']
    if tof_data.bins:
        del tof_data.bins.coords['wavelength']
    return tof_data.rename_dims({'wavelength': 'tof'})


def to_dspacing_with_calibration(
        data: sc.DataArray,
        *,
        calibration: Optional[sc.Dataset] = None) -> sc.DataArray:
    r"""
    Transform coordinates to d-spacing from calibration parameters.

    Computes d-spacing from time-of-flight stored in `data`.
    `data` may have a wavelength coordinate and dimension,
    but those are replaced by tof before conversion to d-spacing.
    An exception is raised if `data` does not contain time-of-flight information.

    :param data: Input data in tof or wavelength dimension.
                 Must have a tof coordinate or attribute.
    :param calibration: Calibration data. If given, use it for the conversion.
                        Otherwise, the calibration data must be stored in `data`.
    :seealso: :func:`ess.diffraction.conversions.dspacing_from_diff_calibration`
    """
    if calibration is not None:
        out = merge_calibration(into=data, calibration=calibration)
    else:
        out = data

    out = _restore_tof_if_in_wavelength(out)
    graph = {'dspacing': dspacing_from_diff_calibration}
    out = out.transform_coords('dspacing', graph=graph)

    for name in ('sample_position', 'source_position', 'position'):
        if name in out.coords:
            out.attrs[name] = out.coords.pop(name)
        if out.bins is not None and name in out.bins.coords:
            out.bins.attrs[name] = out.bins.coords.pop(name)

    return out
