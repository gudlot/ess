# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
import pytest
from ess.wfm.choppers import Chopper, ChopperKind
from .common import allclose


@pytest.fixture
def params():
    dim = 'frame'
    return {
        'frequency':
        sc.scalar(56.0, unit=sc.units.one / sc.units.s),
        'phase':
        sc.scalar(0.5, unit='rad'),
        'position':
        sc.vector(value=[0., 0., 5.], unit='m'),
        'opening_angles_center':
        sc.linspace(dim=dim, start=0.25, stop=2.0 * np.pi, num=6, unit='rad'),
        'opening_angles_width':
        sc.linspace(dim=dim, start=0.1, stop=0.6, num=6, unit='rad'),
        'kind':
        ChopperKind.WFM
    }


def test_chopper_constructor(params):
    chopper = Chopper(**params)
    assert sc.identical(chopper.frequency, params['frequency'])
    assert sc.identical(chopper.phase, params['phase'])
    assert sc.identical(chopper.position, params['position'])
    assert sc.identical(chopper.opening_angles_center, params['opening_angles_center'])
    assert sc.identical(chopper.opening_angles_width, params['opening_angles_width'])
    assert chopper.kind == params['kind']


def test_chopper_constructor_deg():
    dim = 'frame'
    frequency = sc.scalar(56.0, unit=sc.units.one / sc.units.s)
    phase = sc.scalar(15.0, unit='deg')
    position = sc.vector(value=[0., 0., 5.], unit='m')
    opening_angles_center = sc.linspace(dim=dim,
                                        start=60.0,
                                        stop=360.0,
                                        num=6,
                                        unit='deg')
    opening_angles_width = sc.linspace(dim=dim, start=5.0, stop=20.0, num=6, unit='deg')
    kind = ChopperKind.WFM
    chopper = Chopper(frequency=frequency,
                      phase=phase,
                      position=position,
                      opening_angles_center=opening_angles_center,
                      opening_angles_width=opening_angles_width,
                      kind=kind)
    assert sc.identical(chopper.frequency, frequency)
    assert sc.isclose(sc.to_unit(chopper.phase, 'deg'), phase).value
    assert sc.identical(chopper.position, position)
    assert allclose(sc.to_unit(chopper.opening_angles_center, 'deg'),
                    opening_angles_center)
    assert allclose(sc.to_unit(chopper.opening_angles_width, 'deg'),
                    opening_angles_width)
    assert chopper.kind == kind


def test_chopper_constructor_from_open_close_angles(params):
    opening_angles_open = params[
        'opening_angles_center'] - 0.5 * params['opening_angles_width']
    opening_angles_close = params[
        'opening_angles_center'] + 0.5 * params['opening_angles_width']
    chopper = Chopper(frequency=params['frequency'],
                      phase=params['phase'],
                      position=params['position'],
                      opening_angles_open=opening_angles_open,
                      opening_angles_close=opening_angles_close,
                      kind=params['kind'])
    assert allclose(chopper.opening_angles_center, params['opening_angles_center'])
    assert allclose(chopper.opening_angles_width, params['opening_angles_width'])


def test_chopper_constructor_bad_widths(params):
    params['opening_angles_width'].values[1] = -3.0
    with pytest.raises(ValueError) as e_info:
        Chopper(**params)
    assert str(e_info.value) == "Negative window width found in chopper opening angles."


def test_chopper_constructor_bad_centers(params):
    params['opening_angles_center'].values = params['opening_angles_center'].values[[
        1, 0, 2, 3, 4, 5
    ]]
    with pytest.raises(ValueError) as e_info:
        Chopper(**params)
    assert str(e_info.value) == "Chopper opening angles are not monotonic."


def test_chopper_constructor_bad_open_angles(params):
    opening_angles_open = params[
        'opening_angles_center'] - 0.5 * params['opening_angles_width']
    opening_angles_close = params[
        'opening_angles_center'] + 0.5 * params['opening_angles_width']
    opening_angles_open.values = opening_angles_open.values[[1, 0, 2, 3, 4, 5]]
    with pytest.raises(ValueError) as e_info:
        Chopper(frequency=params['frequency'],
                phase=params['phase'],
                position=params['position'],
                opening_angles_open=opening_angles_open,
                opening_angles_close=opening_angles_close,
                kind=params['kind'])
    # This will raise the error on the widths before it reaches the monotonicity check
    assert str(e_info.value) == "Negative window width found in chopper opening angles."


def test_chopper_constructor_bad_open_and_close_angles(params):
    opening_angles_open = params[
        'opening_angles_center'] - 0.5 * params['opening_angles_width']
    opening_angles_close = params[
        'opening_angles_center'] + 0.5 * params['opening_angles_width']
    opening_angles_open.values = opening_angles_open.values[[1, 0, 2, 3, 4, 5]]
    opening_angles_close.values = opening_angles_close.values[[1, 0, 2, 3, 4, 5]]
    with pytest.raises(ValueError) as e_info:
        Chopper(frequency=params['frequency'],
                phase=params['phase'],
                position=params['position'],
                opening_angles_open=opening_angles_open,
                opening_angles_close=opening_angles_close,
                kind=params['kind'])
    assert str(e_info.value) == "Chopper opening angles are not monotonic."


def test_chopper_constructor_bad_close_angles(params):
    dims = params['opening_angles_center'].dims
    with pytest.raises(ValueError) as e_info:
        Chopper(frequency=params['frequency'],
                phase=params['phase'],
                position=params['position'],
                opening_angles_open=sc.array(dims=dims,
                                             values=[0.0, 1.0, 2.0],
                                             unit='rad'),
                opening_angles_close=sc.array(dims=dims,
                                              values=[4.0, 3.0, 5.0],
                                              unit='rad'),
                kind=params['kind'])
    assert str(e_info.value) == "Chopper closing angles are not monotonic."


def test_chopper_constructor_bad_lengths(params):
    params['opening_angles_center'].values = params['opening_angles_center'].values[[
        1, 0, 2, 3, 4, 5
    ]]
    with pytest.raises(ValueError) as e_info:
        Chopper(frequency=params['frequency'],
                phase=params['phase'],
                position=params['position'],
                opening_angles_center=params['opening_angles_center'][
                    params['opening_angles_center'].dims[0], :-1],
                opening_angles_width=params['opening_angles_width'],
                kind=params['kind'])
    assert str(
        e_info.value) == ("All angle input arrays (centers, widths, open or close) "
                          "must have the same length.")


def test_angular_frequency(params):
    chopper = Chopper(**params)
    assert sc.identical(chopper.angular_frequency,
                        (2.0 * np.pi * sc.units.rad) * params['frequency'])


def test_frame_time_open(params):
    dim = 'frame'
    chopper = Chopper(
        frequency=sc.scalar(0.5, unit=sc.units.one / sc.units.s),
        phase=sc.scalar(0., unit='rad'),
        position=params['position'],
        opening_angles_open=sc.array(dims=[dim],
                                     values=np.pi * np.array([0.0, 0.5, 1.0]),
                                     unit='rad'),
        opening_angles_close=sc.array(dims=[dim],
                                      values=np.pi * np.array([0.5, 1.0, 1.5]),
                                      unit='rad'),
        kind=params['kind'])

    assert allclose(
        chopper.time_open,
        sc.to_unit(sc.array(dims=[dim], values=[0.0, 0.5, 1.0], unit='s'), 'us'))
    assert allclose(
        chopper.time_close,
        sc.to_unit(sc.array(dims=[dim], values=[0.5, 1.0, 1.5], unit='s'), 'us'))

    chopper.phase = sc.scalar(2.0 * np.pi / 3.0, unit='rad')
    assert allclose(
        chopper.time_open,
        sc.to_unit(
            sc.array(dims=[dim], values=np.array([0.0, 0.5, 1.0]) + 2.0 / 3.0,
                     unit='s'), 'us'))
    assert allclose(
        chopper.time_close,
        sc.to_unit(
            sc.array(dims=[dim], values=np.array([0.5, 1.0, 1.5]) + 2.0 / 3.0,
                     unit='s'), 'us'))
