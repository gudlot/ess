# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

import scipp as sc
from ess.sans import normalization


def test_solid_angle():
    da = sc.DataArray(data=sc.arange('x', start=1., stop=10., unit='m'),
                      coords={'L2': sc.arange('x', start=1., stop=10., unit='m')})
    pixel_width = 0.5
    pixel_height = 0.75
    expected = (pixel_width * pixel_height) / da.coords['L2']**2
    assert sc.allclose(
        normalization.solid_angle(data=da,
                                  pixel_width=pixel_width,
                                  pixel_height=pixel_height), expected)


def test_transmission_fraction():
    data_incident_monitor = sc.scalar(2., unit='counts')
    data_transmission_monitor = sc.scalar(3., unit='counts')
    direct_incident_monitor = sc.scalar(4., unit='counts')
    direct_transmission_monitor = sc.scalar(5., unit='counts')
    expected = (data_transmission_monitor / direct_transmission_monitor) * (
        direct_incident_monitor / data_incident_monitor)
    result = normalization.transmission_fraction(
        data_incident_monitor=data_incident_monitor,
        data_transmission_monitor=data_transmission_monitor,
        direct_incident_monitor=direct_incident_monitor,
        direct_transmission_monitor=direct_transmission_monitor)
    assert sc.allclose(result, expected)
