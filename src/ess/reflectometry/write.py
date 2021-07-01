# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# flake8: noqa: E501
"""
Functions for file writing
"""

# author: Andrew R. McCluskey (arm61)

import copy
import numpy as np
from ess.reflectometry import orso


def reflectometry(data, filename, bins, header=None):
    """
    Write the reflectometry intensity data to a file.

    :param filename: The file path for the file to be saved to
    :type filename: str
    :param bins: bin edges for qz
    :type bins: scipp._scipp.core.Variable
    :param header: ORSO-compatible header object
    :type: ess.reflectometry.Orso
    """
    binned = data.q_bin(bins)
    q_z_edges = binned.coords["qz"].values
    q_z_vector = q_z_edges[:-1] + np.diff(q_z_edges)
    dq_z_vector = binned.coords["sigma_qz_by_qz"].values
    intensity = binned.data.values
    dintensity = np.sqrt(binned.data.variances)
    if header is None:
        header = str(data.orso)
    np.savetxt(filename,
               np.array([q_z_vector, intensity, dintensity, dq_z_vector]).T,
               fmt='%.16e',
               header=str(header))


def wavelength_theta(data, filename, bins, header=None):
    """
    Write the reflectometry intensity data as a function of wavelength-theta to a file.

    Args:
        filename (:py:attr:`str`): The file path for the file to be saved to.
        bins (:py:attr:`tuple` of :py:attr:`array_like`): wavelength and theta edges.
        header (:py:class:`ess.reflectometry.Orso`): ORSO-compatible header object.
    """
    try:
        binned = data.sample.wavelength_theta_bin(bins).bins.sum(
        ) / data.reference.wavelength_theta_bin(bins).bins.sum()
    except AttributeError:
        binned = data.wavelength_theta_bin(bins).bins.sum()
    theta_c = binned.coords['theta'].values[:-1] + np.diff(
        binned.coords['theta'].values)
    wavelength_c = binned.coords['wavelength'].values[:-1] + np.diff(
        binned.coords['wavelength'].values)
    if header is None:
        new_orso = copy.copy(data.orso)
    else:
        new_orso = copy.copy(header)
    c1 = orso.Column('wavelength', str(binned.coords['wavelength'].unit))
    c2 = orso.Column('theta', str(binned.coords['theta'].unit))
    c3 = orso.Column(
        'Reflectivity', 'dimensionless',
        'A 2D map with theta in horizontal and wavelength in vertical')
    new_orso.columns = [c1, c2, c3]
    out_array = np.zeros((binned.shape[0] + 2, binned.shape[1]))
    out_array[0] = wavelength_c
    out_array[1] = theta_c
    out_array[2:] = binned.values
    np.savetxt(filename, out_array.T, fmt='%.16e', header=str(new_orso))
