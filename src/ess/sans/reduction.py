# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
import scippneutron as scn

def simple_reducer(*, dim):
    """
    Helper function for full range reduction
    """
    return lambda x: sc.sum(x, dim=dim)


def grouping_reducer(*, dim, group):
    """
    Helper function for wavelength slices
    """
    return lambda x: sc.groupby(x, group=group).sum(dim=dim)


def reduce_to_q(data, *, q_bins, reducer, wavelength_bands=None):
    """
    Example:
    >>> reduced = reduce_to_q(data, q_bins=q_bins, reducer=simple_reducer('spectrum'))  # noqa: E501
    """
    wavelength = data.coords['wavelength']
    data = scn.convert(data, 'wavelength', 'Q', scatter=True)
    if wavelength_bands is None:
        data = sc.histogram(data, q_bins)
        return reducer(data)

    wavelength.rename_dims({'wavelength': 'Q'})
    data.coords['wavelength'] = wavelength
    bands = None

    for i in range(wavelength_bands.sizes['wavelength'] - 1):
        low = wavelength_bands['wavelength', i]
        high = wavelength_bands['wavelength', i + 1]
        band = sc.histogram(data['wavelength', low:high].copy(), q_bins)
        band = reducer(band)
        bands = sc.concatenate(bands, band,
                               'wavelength') if bands is not None else band

    bands.coords['wavelength'] = wavelength_bands
    return bands
