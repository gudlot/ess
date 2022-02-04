# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import pytest

from ess.diffraction import load_calibration
from ess.powgen import data


def test_load_calibration_loads_required_data():
    loaded = load_calibration(data.calibration_file(),
                              instrument_filename='POWGEN_Definition_2011-02-25.xml')

    assert 'difa' in loaded
    assert 'difc' in loaded
    assert 'tzero' in loaded
    assert 'mask' in loaded
    assert 'detector' in loaded.coords
    assert loaded.dims == ['detector']


def test_load_calibration_requires_instrument_definition():
    with pytest.raises(ValueError):
        load_calibration(data.calibration_file())


def test_load_calibration_can_only_have_1_instrument_definition():
    with pytest.raises(ValueError):
        load_calibration(data.calibration_file(),
                         instrument_name='POWGEN',
                         instrument_filename='POWGEN_Definition_2011-02-25.xml')
