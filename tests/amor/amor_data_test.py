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
                               reduction_creator_affiliation='ess',
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
        assert_equal(p.orso.creator.affiliation, 'ess')
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

    # Commented out until the reference.nxs file has a home
    # def test_normalisation_data_file(self):
    #     p = amor_data.AmorData(BINNED.copy())
    #     file_path = (os.path.dirname(os.path.realpath(__file__)) +
    #                  os.path.sep + "reference.nxs")
    #     q = amor_data.AmorReference(BINNED.copy(), data_file=file_path)
    #     z = amor_data.Normalisation(p, q)
    #     assert_equal(
    #         z.sample.orso.reduction.input_files.reference_files[0].file,
    #         file_path)

    def test_q_bin_norm(self):
        p = amor_data.AmorData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS.astype(float),
            unit=sc.units.angstrom)
        p.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        q = amor_data.AmorData(BINNED.copy())
        q.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS.astype(float),
            unit=sc.units.angstrom)
        q.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        z = amor_data.Normalisation(p, q)
        del z.sample.event.coords['qz']
        with self.assertRaises(sc.NotFoundError):
            z.q_bin()

    def test_write_reflectometry_norm(self):
        p = amor_data.AmorData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS.astype(float),
            unit=sc.units.angstrom)
        p.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        q = amor_data.AmorData(BINNED.copy())
        q.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS.astype(float),
            unit=sc.units.angstrom)
        q.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        z = amor_data.Normalisation(p, q)
        file_path = (os.path.dirname(os.path.realpath(__file__)) +
                     os.path.sep + "test1.txt")
        z.write_reflectometry(file_path)
        written_data = np.loadtxt(file_path, unpack=True)
        assert_equal(written_data.shape, (4, 199))

    def test_binwavelength_theta_norm(self):
        p = amor_data.AmorData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS.astype(float),
            unit=sc.units.angstrom)
        p.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        q = amor_data.AmorData(BINNED.copy())
        q.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS.astype(float),
            unit=sc.units.angstrom)
        q.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        z = amor_data.Normalisation(p, q)
        bins = np.linspace(0, 100, 10)
        k = z.wavelength_theta_bin((bins, bins))
        assert_equal(k.shape, (9, 9))

    def test_write_wavelength_theta_norm(self):
        p = amor_data.AmorData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS.astype(float),
            unit=sc.units.angstrom)
        p.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        q = amor_data.AmorData(BINNED.copy())
        q.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS.astype(float),
            unit=sc.units.angstrom)
        q.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        z = amor_data.Normalisation(p, q)
        file_path = (os.path.dirname(os.path.realpath(__file__)) +
                     os.path.sep + "test1.txt")
        bins = np.linspace(0, 100, 10)
        z.write_wavelength_theta(file_path, (bins, bins))
        written_data = np.loadtxt(file_path, unpack=True)
        assert_equal(written_data.shape, (11, 9))


class TestQGrid(unittest.TestCase):
    def test_q_grid_a(self):
        actual = amor_data.q_grid(q_fix=0.05)
        expected = [
            0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275,
            0.03, 0.0325, 0.035, 0.0375, 0.04, 0.0425, 0.045, 0.0475, 0.05,
            0.0525, 0.055125, 0.05788125, 0.06077531, 0.06381408, 0.06700478,
            0.07035502, 0.07387277, 0.07756641, 0.08144473
        ]
        assert_almost_equal(actual, expected)

    def test_q_grid_b(self):
        actual = amor_data.q_grid(q_min=0.07, q_fix=0.1)
        expected = [0.07, 0.075, 0.08, 0.085]
        assert_almost_equal(actual, expected)

    def test_q_grid_c(self):
        actual = amor_data.q_grid(q_min=0.07, q_fix=0.001)
        expected = [0.06641707, 0.06973792, 0.07322482, 0.07688606, 0.08073037]
        assert_almost_equal(actual, expected)

    def test_q_grid_d(self):
        actual = amor_data.q_grid(d_q=0.0005)
        expected = [
            0.0073205, 0.00805255, 0.00885781, 0.00974359, 0.01071794,
            0.01178974, 0.01296871, 0.01426558, 0.01569214, 0.01726136,
            0.01898749, 0.02088624, 0.02297486, 0.02527235, 0.02779959,
            0.03057955, 0.0336375, 0.03700125, 0.04070137, 0.04477151,
            0.04924866, 0.05417353, 0.05959088, 0.06554997, 0.07210497,
            0.07931546
        ]
        assert_almost_equal(actual, expected)

    def test_q_grid_e(self):
        with self.assertRaises(ValueError):
            _ = amor_data.q_grid(q_min=0.7, q_max=0.001)
