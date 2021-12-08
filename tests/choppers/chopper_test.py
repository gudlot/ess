# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
import pytest
import ess.choppers as ch


@pytest.fixture
def params():
    dim = 'frame'
    return {
        'frequency':
        sc.scalar(56.0, unit="Hz"),
        'phase':
        sc.scalar(0.5, unit='rad'),
        'position':
        sc.vector(value=[0., 0., 5.], unit='m'),
        'cutout_angles_center':
        sc.linspace(dim=dim, start=0.25, stop=2.0 * np.pi, num=6, unit='rad'),
        'cutout_angles_width':
        sc.linspace(dim=dim, start=0.1, stop=0.6, num=6, unit='rad'),
        'kind':
        sc.scalar('wfm')
    }


def test_angular_frequency(params):
    chopper = sc.Dataset(data=params)
    assert sc.identical(ch.angular_frequency(chopper),
                        (2.0 * np.pi * sc.units.rad) * params['frequency'])


def test_cutout_angles(params):
    dim = 'frame'
    chopper = sc.Dataset(data=params)
    centers = sc.linspace(dim=dim, start=0.25, stop=2.0 * np.pi, num=6, unit='rad')
    widths = sc.linspace(dim=dim, start=0.1, stop=0.6, num=6, unit='rad')
    assert sc.allclose(ch.cutout_angles_begin(chopper), centers - 0.5 * widths)
    assert sc.allclose(ch.cutout_angles_end(chopper), centers + 0.5 * widths)


def test_time_open_closed(params):
    dim = 'frame'
    chopper = sc.Dataset(
        data={
            "frequency":
            sc.scalar(0.5, unit=sc.units.one / sc.units.s),
            "phase":
            sc.scalar(0., unit='rad'),
            "position":
            params['position'],
            "cutout_angles_begin":
            sc.array(dims=[dim], values=np.pi * np.array([0.0, 0.5, 1.0]), unit='rad'),
            "cutout_angles_end":
            sc.array(dims=[dim], values=np.pi * np.array([0.5, 1.0, 1.5]), unit='rad'),
            "kind":
            params['kind']
        })

    assert sc.allclose(
        ch.time_open(chopper),
        sc.to_unit(sc.array(dims=[dim], values=[0.0, 0.5, 1.0], unit='s'), 'us'))
    assert sc.allclose(
        ch.time_closed(chopper),
        sc.to_unit(sc.array(dims=[dim], values=[0.5, 1.0, 1.5], unit='s'), 'us'))

    chopper["phase"] = sc.scalar(2.0 * np.pi / 3.0, unit='rad')
    assert sc.allclose(
        ch.time_open(chopper),
        sc.to_unit(
            sc.array(dims=[dim], values=np.array([0.0, 0.5, 1.0]) + 2.0 / 3.0,
                     unit='s'), 'us'))
    assert sc.allclose(
        ch.time_closed(chopper),
        sc.to_unit(
            sc.array(dims=[dim], values=np.array([0.5, 1.0, 1.5]) + 2.0 / 3.0,
                     unit='s'), 'us'))


def test_find_chopper_keys():
    da = sc.DataArray(data=sc.scalar('dummy'),
                      coords={
                          'chopper3': sc.scalar(0),
                          'abc': sc.scalar(0),
                          'chopper_1': sc.scalar(0),
                          'sample': sc.scalar(0),
                          'source': sc.scalar(0),
                          'Chopper_wfm': sc.scalar(0),
                          'chopper0': sc.scalar(0),
                          'chopper5': sc.scalar(0),
                          'monitor': sc.scalar(0)
                      })
    expected = ['chopper3', 'chopper_1', 'Chopper_wfm', 'chopper0', 'chopper5']
    assert ch.find_chopper_keys(da) == expected
