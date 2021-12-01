# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
import scippneutron as scn


def instrument_view(da: sc.DataArray,
                    components: dict = None,
                    pixel_size: float = 0.0035,
                    **kwargs) -> sc.plotting.objects.Plot:
    """
    Instrument view for the Amor instrument, which automatically populates a list of
    additional beamline components, and sets the pixel size.

    :param da: The input data for which to display the instrument view.
    :param components: A dict of additional components to display. If `None`, the
        sample and the source chopper are automatically added. Default is `None`.
    :param pixel_size: The detector pixel size. Default is 0.0035.
    """
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
