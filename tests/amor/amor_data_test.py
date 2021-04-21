# flake8: noqa: E501
"""
Tests for amor_data module
"""

# author: Andrew R. McCluskey (arm61)

import os
import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import scipp as sc
from ess.amor import amor_data

np.random.seed(1)

N = 9
VALUES = np.ones(N)
DETECTORS = np.random.randint(1, 5, size=(N))

DATA = sc.DataArray(
    data=sc.Variable(
        dims=["event"],
        unit=sc.units.counts,
        values=VALUES,
        dtype=sc.dtype.float32,
    ),
    coords={
        "detector_id":
        sc.Variable(dims=["event"], values=DETECTORS, dtype=sc.dtype.int32),
    },
)

DETECTOR_ID = sc.Variable(dims=["detector_id"],
                          values=np.arange(1, 5),
                          dtype=sc.dtype.int32)
BINNED = sc.bin(DATA, groups=[DETECTOR_ID])

PIXELS = np.array([[1, 1, 1], [1, 2, 1], [2, 1, 1], [2, 2, 1]])
X = sc.Variable(
    dims=["detector_id"],
    values=PIXELS[:, 0],
    dtype=sc.dtype.float64,
    unit=sc.units.m,
)
Y = sc.Variable(
    dims=["detector_id"],
    values=PIXELS[:, 1],
    dtype=sc.dtype.float64,
    unit=sc.units.m,
)
Z = sc.Variable(
    dims=["detector_id"],
    values=PIXELS[:, 2],
    dtype=sc.dtype.float64,
    unit=sc.units.m,
)
BINNED.coords["position"] = sc.geometry.position(X, Y, Z)
BINNED.bins.constituents['data'].coords["tof"] = sc.Variable(
    dims=["event"],
    values=np.linspace(1, 10, N),
    unit=sc.units.us,
)
BINNED.attrs['sample_position'] = sc.geometry.position(0. * sc.units.m,
                                                       0. * sc.units.m,
                                                       0. * sc.units.m)
BINNED.attrs['instrument_name'] = sc.scalar(value='AMOR')
BINNED.attrs['experiment_title'] = sc.scalar(value='test')


class TestAmorData(unittest.TestCase):
    def test_amordata_init(self):
        """
        Testing the default initialisation of the AmorData objects.
        """
        p = amor_data.AmorData(BINNED.copy())
        assert_equal(isinstance(p.data, sc._scipp.core.DataArray), True)
        assert_equal(isinstance(p.data.data, sc._scipp.core.VariableView),
                     True)
        assert_almost_equal(
            np.sort(p.data.bins.constituents["data"].coords["tof"].values),
            [
                71667.6666667, 71668.7916667, 71669.9166667, 71671.0416667,
                71672.1666667, 71673.2916667, 71674.4166667, 71675.5416667,
                71676.6666667
            ],
        )
        assert_almost_equal(
            p.data.attrs['source_position'].values,
            sc.geometry.position(0. * sc.units.m, 0. * sc.units.m,
                                 -15. * sc.units.m).values)
        assert_almost_equal(p.data.coords['sigma_lambda_by_lambda'].values,
                            [0.030625, 0.030625, 0.030625, 0.030625])
        assert_almost_equal(p.tau.value, 75000)
        assert_almost_equal(p.chopper_detector_distance.value, 19e10)
        assert_almost_equal(p.chopper_chopper_distance.value, 0.49)
        assert_almost_equal(p.chopper_phase.value, -8)
        assert_almost_equal(p.wavelength_cut.value, 2.4)

    def test_amordata_init_nodefault(self):
        """
        Testing the default initialisation of the AmorData objects.
        """
        p = amor_data.AmorData(BINNED.copy(),
                               reduction_creator='andrew',
                               data_owner='andrew',
                               experiment_id='1234',
                               experiment_date='2021-04-21',
                               sample_description='my sample',
                               reduction_file='my_notebook.ipynb',
                               chopper_phase=-5 * sc.units.deg,
                               chopper_chopper_distance=0.3 * sc.units.m,
                               chopper_detector_distance=18e10 *
                               sc.units.angstrom,
                               wavelength_cut=2.0 * sc.units.angstrom)
        assert_equal(isinstance(p.data, sc._scipp.core.DataArray), True)
        assert_equal(isinstance(p.data.data, sc._scipp.core.VariableView),
                     True)
        assert_almost_equal(
            np.sort(p.data.bins.constituents["data"].coords["tof"].values),
            [
                72917.6666667, 72918.7916667, 72919.9166667, 72921.0416667,
                72922.1666667, 72923.2916667, 72924.4166667, 72925.5416667,
                72926.6666667
            ],
        )
        assert_almost_equal(
            p.data.attrs['source_position'].values,
            sc.geometry.position(0. * sc.units.m, 0. * sc.units.m,
                                 -15. * sc.units.m).values)
        assert_almost_equal(p.data.coords['sigma_lambda_by_lambda'].values,
                            [0.01875, 0.01875, 0.01875, 0.01875])
        assert_almost_equal(p.tau.value, 75000)
        assert_almost_equal(p.chopper_detector_distance.value, 18e10)
        assert_almost_equal(p.chopper_chopper_distance.value, 0.3)
        assert_almost_equal(p.chopper_phase.value, -5)
        assert_almost_equal(p.wavelength_cut.value, 2.)
        assert_equal(p.orso.creator.name, 'andrew')
        assert_equal(p.orso.data_source.owner, 'andrew')
        assert_equal(p.orso.data_source.experiment_id, '1234')
        assert_equal(p.orso.data_source.experiment_date, '2021-04-21')
        assert_equal(p.orso.reduction.script, 'my_notebook.ipynb')

    def test_wavelength_masking(self):
        p = amor_data.AmorData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.angstrom,
        )
        p.wavelength_masking(
            wavelength_min=2 * sc.units.angstrom,
            wavelength_max=4 * sc.units.angstrom,
        )
        assert_equal(
            p.event.masks["wavelength"].values,
            ~((DETECTORS >= 2) & (DETECTORS <= 4)),
        )

    def test_wavelength_masking_no_min(self):
        p = amor_data.AmorData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.angstrom,
        )
        p.wavelength_masking(wavelength_max=4 * sc.units.angstrom)
        assert_equal(
            p.event.masks["wavelength"].values,
            ~((DETECTORS >= 2.4) & (DETECTORS <= 4)),
        )

    def test_wavelength_masking_no_max(self):
        p = amor_data.AmorData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.angstrom,
        )
        p.wavelength_masking(wavelength_min=2 * sc.units.angstrom)
        assert_equal(
            p.event.masks["wavelength"].values,
            ~((DETECTORS >= 2) & (DETECTORS <= 5)),
        )


class TestAmorReference(unittest.TestCase):
    def test_amorreference_init(self):
        """
        Testing the default initialisation of the AmorReference objects.
        """
        p = amor_data.AmorReference(BINNED.copy())
        assert_equal(isinstance(p.data, sc._scipp.core.DataArray), True)
        assert_equal(isinstance(p.data.data, sc._scipp.core.VariableView),
                     True)
        assert_equal(p.m_value.value, 5)
        assert_almost_equal(p.event.coords['normalisation'].values, np.ones(9))

    def test_amorreference_init_nodefault(self):
        """
        Testing the default initialisation of the AmorReference objects.
        """
        p = amor_data.AmorReference(BINNED.copy(),
                                    m_value=4 * sc.units.dimensionless)
        assert_equal(isinstance(p.data, sc._scipp.core.DataArray), True)
        assert_equal(isinstance(p.data.data, sc._scipp.core.VariableView),
                     True)
        assert_equal(p.m_value.value, 4)
        assert_almost_equal(p.event.coords['normalisation'].values, np.ones(9))


class TestNormalisation(unittest.TestCase):
    def test_normalisation_init(self):
        p = amor_data.AmorData(BINNED.copy())
        q = amor_data.AmorReference(BINNED.copy())
        z = amor_data.Normalisation(p, q)
        assert_almost_equal(z.sample.data.bins.constituents['data'].values,
                            p.data.data.bins.constituents['data'].values)
