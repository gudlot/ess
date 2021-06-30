# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)


def reduce_by_wavelength(data, q_bins, groupby, wavelength_bands):
    # TODO Backup of the coord is necessary until `convert` can keep original
    wavelength = data.coords['wavelength']
    data = sc.neutron.convert(data, 'wavelength', 'Q',
                              out=data)  # TODO no gravity yet
    data.coords['wavelength'] = wavelength
    bands = None
    for i in range(wavelength_bands.sizes['wavelength'] - 1):
        low = wavelength_bands['wavelength', i]
        high = wavelength_bands['wavelength', i + 1]
        band = sc.histogram(data['wavelength', low:high], q_bins)
        band = sc.groupby(band, group=groupby).sum('spectrum')
        bands = sc.concatenate(bands, band,
                               'wavelength') if bands is not None else band
    bands.coords['wavelength'] = wavelength_bands
    return bands
