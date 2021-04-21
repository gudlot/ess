# flake8: noqa: E501
"""
Functions for file writing
"""

# author: Andrew R. McCluskey (arm61)

import copy
import numpy as np
from ess.amor import amor_data
from ess.reflectometry import orso


def reflectometry(data, filename, bin_kwargs=None, header=None):
    """
    Write the reflectometry intensity data to a file.

    Args:
        filename (`str`): The file path for the file to be saved to.
        bin_kwargs (`dict`, optional): A dictionary of keyword arguments to be passed to the :py:func:`q_bin` class method. Optional, default is that default :py:func:`q_bin` keywords arguments are used.
        header (`ess.reflectometry.Orso`): ORSO-compatible header object.
    """
    if bin_kwargs is None:
        binned = data.q_bin()
    else:
        binned = data.q_bin(**bin_kwargs)
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
        filename (`str`): The file path for the file to be saved to.
        bins (`tuple` of `array_like`): wavelength and theta edges.
        header (`ess.reflectometry.Orso`): ORSO-compatible header object.
    """
    if isinstance(data, amor_data.Normalisation):
        binned = data.sample.wavelength_theta_bin(bins).bins.sum(
        ) / data.reference.wavelength_theta_bin(bins).bins.sum()
    else:
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
