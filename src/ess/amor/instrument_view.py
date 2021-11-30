# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
import scippneutron as scn


def instrument_view(da, components=None, pixel_size=0.0035, **kwargs):
    if components is None:
        components = {
            "sample": {
                'center': da.meta['sample_position'],
                'color': 'red',
                'size': sc.vector(value=[0.2, 0.01, 0.2], unit=sc.units.m),
                'type': 'box'
            },
            "source_chopper": {
                'center': da.meta['source_chopper'].value.position,
                'color': 'blue',
                'size': sc.vector(value=[0.5, 0, 0], unit=sc.units.m),
                'type': 'disk'
            }
        }

    return scn.instrument_view(da,
                               components=components,
                               pixel_size=pixel_size,
                               **kwargs)
