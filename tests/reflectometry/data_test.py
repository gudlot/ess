# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# flake8: noqa: E501
"""
Tests for data module
"""

# author: Andrew R. McCluskey (arm61)

import os
import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import scipp as sc
from ess.reflectometry import data
from ..tools.io import file_location

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
        "detector_id": sc.Variable(dims=["event"],
                                   values=DETECTORS,
                                   dtype=sc.dtype.int32)
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

BINNED.attrs['instrument_name'] = sc.scalar(value='AMOR')
BINNED.attrs['experiment_title'] = sc.scalar(value='test')


class TestData(unittest.TestCase):
    # Commented out until the sample.nxs file has a home
    # def test_refldata_file(self):
    #     file_path = (os.path.dirname(os.path.realpath(__file__)) +
    #                  os.path.sep + "sample.nxs")
    #     p = data.ReflData(file_path)
    #     assert_equal(isinstance(p.data, sc._scipp.core.DataArray), True)
    #     assert_equal(p.data_file, file_path)

    def test_refldata_init(self):
        """
        Testing the default initialisation of the ReflData objects.
        """
        p = data.ReflData(BINNED.copy())
        assert_equal(isinstance(p.data, sc._scipp.core.DataArray), True)
        assert_equal(isinstance(p.data.data, sc._scipp.core.Variable), True)
        assert_almost_equal(p.data.coords["position"].fields.x.values, X.values)
        assert_almost_equal(p.data.coords["position"].fields.y.values, Y.values)
        assert_almost_equal(p.data.coords["position"].fields.z.values, Z.values)
        assert_almost_equal(
            np.sort(p.data.bins.constituents["data"].values),
            np.sort(VALUES),
            decimal=5,
        )
        assert_almost_equal(
            np.sort(p.data.bins.constituents["data"].variances),
            np.sort(np.ones_like(VALUES)),
            decimal=5,
        )
        assert_almost_equal(p.sample_angle_offset.values, 0)
        assert_equal(p.sample_angle_offset.unit, sc.units.deg)

    def test_refldata_init_sample_angle_offset(self):
        """
        Testing the ReflData initialisation with a non-default sample_angle_offset.
        """
        p = data.ReflData(BINNED.copy(), sample_angle_offset=2 * sc.units.deg)
        assert_almost_equal(p.sample_angle_offset.values, 2)
        assert_equal(p.sample_angle_offset.unit, sc.units.deg)

    def test_refldata_event(self):
        p = data.ReflData(BINNED.copy())
        assert_equal(isinstance(p.event, sc._scipp.core.DataArray), True)
        assert_almost_equal(np.sort(p.event.values), np.sort(VALUES), decimal=5)
        assert_almost_equal(
            np.sort(p.event.variances),
            np.sort(np.ones_like(VALUES)),
            decimal=5,
        )

    def test_q_bin(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["qz"] = sc.linspace(dim='event',
                                           start=1,
                                           stop=10,
                                           num=N,
                                           unit=sc.Unit('1/angstrom'))
        p.event.coords["sigma_qz_by_qz"] = sc.linspace(
            dim="event",
            start=0.1,
            stop=1.0,
            num=N,
            unit=sc.Unit('1/angstrom'),
            dtype=sc.dtype.float64,
        )
        p.event.coords["tof"] = sc.Variable(dims=["event"], values=DETECTORS)
        bins = sc.linspace(dim='qz',
                           start=0.,
                           stop=11.,
                           num=4,
                           unit=sc.Unit('1/angstrom'))
        b = p.q_bin(bins)
        assert_almost_equal(b.coords["qz"].values, bins.values)
        assert_almost_equal(b.coords["sigma_qz_by_qz"].values,
                            np.linspace(0.325, 1.0, 3))
        assert_almost_equal(b.data.values, np.array([3.0, 3.0, 3.]) / 9.)
        assert_almost_equal(b.data.variances, np.array([3.0, 3.0, 3.]) / 81.)

    def test_q_bin_different_unit(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["qz"] = sc.linspace(dim="event",
                                           start=1,
                                           stop=10,
                                           num=N,
                                           unit=(1 / sc.units.m).unit)
        p.event.coords["sigma_qz_by_qz"] = sc.linspace(dim="event",
                                                       start=0.1,
                                                       stop=1.0,
                                                       num=N,
                                                       unit=(1 / sc.units.m).unit,
                                                       dtype=sc.dtype.float64)
        p.event.coords["tof"] = sc.Variable(dims=["event"], values=DETECTORS)
        bins = sc.linspace(dim='qz', start=0, stop=11, num=4, unit=sc.Unit('1/m'))
        b = p.q_bin(bins)
        assert_almost_equal(b.coords["qz"].values, bins.values)
        assert_almost_equal(b.coords["sigma_qz_by_qz"].values,
                            np.linspace(0.325, 1.0, 3))
        assert_almost_equal(b.data.values, np.array([3.0, 3.0, 3.]) / 9.)
        assert_almost_equal(b.data.variances, np.array([3.0, 3.0, 3.]) / 81.)

    def test_q_bin_no_qz(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["sigma_qz_by_qz"] = sc.linspace(dim="event",
                                                       start=0.1,
                                                       stop=1.0,
                                                       num=N,
                                                       unit=(1 / sc.units.m).unit,
                                                       dtype=sc.dtype.float64)
        p.event.coords["tof"] = sc.Variable(dims=["event"], values=DETECTORS)
        bins = sc.linspace(dim='qz',
                           start=0,
                           stop=11,
                           num=4,
                           unit=sc.Unit('1/angstrom'))
        with self.assertRaises(sc.NotFoundError):
            _ = p.q_bin(bins)

    def test_q_bin_no_qzresolution(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["qz"] = sc.linspace(dim="event",
                                           start=1,
                                           stop=10,
                                           num=N,
                                           unit=sc.Unit('1/angstrom'))
        p.event.coords["tof"] = sc.Variable(dims=["event"], values=DETECTORS)
        bins = sc.linspace(dim='qz',
                           start=0,
                           stop=11,
                           num=4,
                           unit=sc.Unit('1/angstrom'))
        b = p.q_bin(bins)
        assert_almost_equal(b.coords["qz"].values, bins.values)
        assert_almost_equal(b.data.values, np.array([3.0, 3.0, 3.]) / 9.)
        assert_almost_equal(b.data.variances, np.array([3.0, 3.0, 3.]) / 81.)

    def test_wavelength_theta_bin(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.angstrom,
        )
        p.event.coords["theta"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.deg,
        )
        bins1 = sc.linspace(dim='wavelength',
                            start=0.01,
                            stop=2,
                            num=50,
                            unit=sc.Unit('angstrom'))
        bins2 = sc.linspace(dim='theta',
                            start=0.01,
                            stop=2,
                            num=50,
                            unit=sc.Unit('deg'))
        binned = p.wavelength_theta_bin((bins1, bins2))
        assert_equal(binned.shape, [49, 49])

    def test_q_theta_bin(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["qz"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.Unit('1/angstrom'),
        )
        p.event.coords["theta"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.deg,
        )
        bins1 = sc.linspace(dim='qz',
                            start=0.01,
                            stop=2,
                            num=50,
                            unit=sc.Unit('1/angstrom'))
        bins2 = sc.linspace(dim='theta',
                            start=0.01,
                            stop=2,
                            num=50,
                            unit=sc.Unit('deg'))
        binned = p.q_theta_bin((bins1, bins2))
        assert_equal(binned.shape, [49, 49])

    def test_wavelength_q_bin(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["qz"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.Unit('1/angstrom'),
        )
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.angstrom,
        )
        bins1 = sc.linspace(dim='wavelength',
                            start=0.01,
                            stop=2,
                            num=50,
                            unit=sc.Unit('angstrom'))
        bins2 = sc.linspace(dim='qz',
                            start=0.01,
                            stop=2,
                            num=50,
                            unit=sc.Unit('1/angstrom'))
        binned = p.wavelength_q_bin((bins1, bins2))
        assert_equal(binned.shape, [49, 49])

    def test_tof_to_wavelength(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["tof"] = sc.Variable(dims=["event"],
                                            values=DETECTORS,
                                            dtype=sc.dtype.float64)
        p.data.attrs["source_position"] = sc.geometry.position(
            0.0 * sc.units.m, 0.0 * sc.units.m, -15.0 * sc.units.m)
        p.data.attrs["sample_position"] = sc.geometry.position(
            0.0 * sc.units.m, 0.0 * sc.units.m, 0.0 * sc.units.m)
        p.find_wavelength()
        assert_almost_equal(
            p.event.coords["wavelength"].values,
            [
                0.0004729,
                0.0009457,
                0.0002267,
                0.0002267,
                0.0009069,
                0.0004396,
                0.0008791,
                0.0004396,
                0.0008791,
            ],
        )

    def test_find_theta_gravity(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.angstrom,
        )
        p.data.attrs["sample_position"] = sc.geometry.position(
            0.0 * sc.units.m, 0.0 * sc.units.m, 0.0 * sc.units.m)
        p.find_theta()
        assert_almost_equal(
            p.event.coords["theta"].values,
            [
                44.9999641, 44.9998564, 63.4349452, 63.4349452, 63.4348914, 63.4349345,
                63.4348914, 63.4349345, 63.4348914
            ],
        )
        assert_almost_equal(
            p.event.coords["sigma_theta_by_theta"].values,
            [
                0.0013517,
                0.0013517,
                0.0009589,
                0.0009589,
                0.0009589,
                0.0009589,
                0.0009589,
                0.0009589,
                0.0009589,
            ],
        )
        assert_almost_equal(
            p.data.attrs["sigma_gamma"].values,
            [0.0608281, 0.0608281, 0.0608281, 0.0608281],
        )

    def test_find_theta_no_gravity(self):
        p = data.ReflData(BINNED.copy(), gravity=False)
        with self.assertRaises(NotImplementedError):
            p.find_theta()

    def test_find_qz(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["theta"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.deg,
        )
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.angstrom,
        )
        p.find_qz()
        assert_almost_equal(
            p.event.coords["qz"].values,
            [
                0.21928,
                0.21914643,
                0.21931341,
                0.21931341,
                0.21914643,
                0.21928,
                0.21914643,
                0.21928,
                0.21914643,
            ],
        )
        assert_almost_equal(p.event.coords["sigma_qz_by_qz"].values, np.zeros(9))

    def test_find_qz_with_resolution(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["theta"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.deg,
        )
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.angstrom,
        )
        p.event.coords["sigma_theta_by_theta"] = sc.Variable(dims=["event"],
                                                             values=DETECTORS * 0.1,
                                                             dtype=sc.dtype.float64)
        p.data.coords["sigma_lambda_by_lamdba"] = sc.Variable(
            dims=["detector_id"],
            values=np.arange(1, 5) * 0.1,
            dtype=sc.dtype.float64,
        )
        p.find_qz()
        assert_almost_equal(
            p.event.coords["qz"].values,
            [
                0.21928,
                0.21914643,
                0.21931341,
                0.21931341,
                0.21914643,
                0.21928,
                0.21914643,
                0.21928,
                0.21914643,
            ],
        )
        assert_almost_equal(
            p.event.coords["sigma_qz_by_qz"].values,
            [
                0.2236068,
                0.4123106,
                0.2236068,
                0.2236068,
                0.4472136,
                0.4472136,
                0.5656854,
                0.4472136,
                0.5656854,
            ],
        )

    def test_illumination(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["theta"] = sc.Variable(
            dims=["event"],
            values=DETECTORS * 0.1,
            dtype=sc.dtype.float64,
            unit=sc.units.deg,
        )
        p.beam_width = 50e-3 * sc.units.m
        p.sample_size = 0.10 * sc.units.m
        p.illumination()
        assert_almost_equal(
            p.event.coords["illumination"].values,
            [
                0.7549526,
                0.9799233,
                0.4389162,
                0.4389162,
                0.9799233,
                0.7549526,
                0.9799233,
                0.7549526,
                0.9799233,
            ],
        )
        assert_almost_equal(
            p.event.data.values,
            1 / np.array(
                [
                    0.7549526,
                    0.9799233,
                    0.4389162,
                    0.4389162,
                    0.9799233,
                    0.7549526,
                    0.9799233,
                    0.7549526,
                    0.9799233,
                ],
                dtype=np.float32,
            ),
        )
        assert_almost_equal(
            p.event.data.variances,
            1 / np.array(
                [
                    0.7549526,
                    0.9799233,
                    0.4389162,
                    0.4389162,
                    0.9799233,
                    0.7549526,
                    0.9799233,
                    0.7549526,
                    0.9799233,
                ],
                dtype=np.float32,
            )**2,
            decimal=6,
        )

    def test_detector_masking_defaults(self):
        p = data.ReflData(BINNED.copy())
        p.detector_masking()
        assert_equal(p.data.masks["x_mask"].values, [False] * 4)
        assert_equal(p.data.masks["y_mask"].values, [False] * 4)
        assert_equal(p.data.masks["z_mask"].values, [False] * 4)

    def test_detector_masking_x_min(self):
        p = data.ReflData(BINNED.copy())
        p.detector_masking(x_min=2 * sc.units.m)
        assert_equal(p.data.masks["x_mask"].values, [True, True, False, False])
        assert_equal(p.data.masks["y_mask"].values, [False] * 4)
        assert_equal(p.data.masks["z_mask"].values, [False] * 4)

    def test_detector_masking_x_max(self):
        p = data.ReflData(BINNED.copy())
        p.detector_masking(x_max=1 * sc.units.m)
        assert_equal(p.data.masks["x_mask"].values, [False, False, True, True])
        assert_equal(p.data.masks["y_mask"].values, [False] * 4)
        assert_equal(p.data.masks["z_mask"].values, [False] * 4)

    def test_detector_masking_y_min(self):
        p = data.ReflData(BINNED.copy())
        p.detector_masking(y_min=2 * sc.units.m)
        assert_equal(p.data.masks["y_mask"].values, [True, False, True, False])
        assert_equal(p.data.masks["x_mask"].values, [False] * 4)
        assert_equal(p.data.masks["z_mask"].values, [False] * 4)

    def test_detector_masking_y_max(self):
        p = data.ReflData(BINNED.copy())
        p.detector_masking(y_max=1 * sc.units.m)
        assert_equal(p.data.masks["y_mask"].values, [False, True, False, True])
        assert_equal(p.data.masks["x_mask"].values, [False] * 4)
        assert_equal(p.data.masks["z_mask"].values, [False] * 4)

    def test_detector_masking_z_min(self):
        p = data.ReflData(BINNED.copy())
        p.detector_masking(z_min=2 * sc.units.m)
        assert_equal(p.data.masks["z_mask"].values, [True, True, True, True])
        assert_equal(p.data.masks["y_mask"].values, [False] * 4)
        assert_equal(p.data.masks["x_mask"].values, [False] * 4)

    def test_detector_masking_z_max(self):
        p = data.ReflData(BINNED.copy())
        p.detector_masking(z_max=1 * sc.units.m)
        assert_equal(p.data.masks["z_mask"].values, [False, False, False, False])
        assert_equal(p.data.masks["y_mask"].values, [False] * 4)
        assert_equal(p.data.masks["x_mask"].values, [False] * 4)

    def test_theta_masking(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["theta"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.deg,
        )
        p.theta_masking(theta_min=2 * sc.units.deg, theta_max=4 * sc.units.deg)
        assert_equal(
            p.data.masks["theta"].values,
            [True, False, True],
        )

    def test_theta_masking_no_min(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["theta"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.deg,
        )
        p.theta_masking(theta_max=4 * sc.units.deg)
        assert_equal(
            p.data.masks["theta"].values,
            [True, False, True],
        )

    def test_theta_masking_no_max(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["theta"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.deg,
        )
        p.theta_masking(theta_min=2 * sc.units.deg)
        assert_equal(
            p.data.masks["theta"].values,
            [True, False, True],
        )

    def test_wavelength_masking(self):
        p = data.ReflData(BINNED.copy())
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
            p.data.masks["wavelength"].values,
            [True, False, True],
        )

    def test_wavelength_masking_no_min(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.angstrom,
        )
        p.wavelength_masking(wavelength_max=4 * sc.units.angstrom)
        assert_equal(
            p.data.masks["wavelength"].values,
            [True, False, True],
        )

    def test_wavelength_masking_no_max(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS,
            dtype=sc.dtype.float64,
            unit=sc.units.angstrom,
        )
        p.wavelength_masking(wavelength_min=2 * sc.units.angstrom)
        assert_equal(
            p.data.masks["wavelength"].values,
            [True, False, True],
        )

    def test_write(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["qz"] = sc.linspace(dim="event",
                                           start=1,
                                           stop=10,
                                           num=N,
                                           unit=sc.Unit('1/angstrom'))
        p.event.coords["sigma_qz_by_qz"] = sc.linspace(dim="event",
                                                       start=0.1,
                                                       stop=1.0,
                                                       num=N,
                                                       unit=sc.Unit('1/angstrom'),
                                                       dtype=sc.dtype.float64)
        p.event.coords["tof"] = sc.Variable(dims=["event"], values=DETECTORS)
        bins = sc.linspace(dim='qz',
                           start=0,
                           stop=11,
                           num=200,
                           unit=sc.Unit('1/angstrom'))
        with file_location("test1.txt") as file_path:
            p.write_reflectometry(file_path, bins)
            written_data = np.loadtxt(file_path, unpack=True)
            assert_equal(written_data.shape, (4, 199))

    def test_write_bins(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["qz"] = sc.linspace(dim="event",
                                           start=1,
                                           stop=10,
                                           num=N,
                                           unit=sc.Unit('1/angstrom'))
        p.event.coords["sigma_qz_by_qz"] = sc.linspace(dim="event",
                                                       start=0.1,
                                                       stop=1.0,
                                                       num=N,
                                                       unit=sc.Unit('1/angstrom'),
                                                       dtype=sc.dtype.float64)
        p.event.coords["tof"] = sc.Variable(dims=["event"], values=DETECTORS)
        bins = sc.linspace(dim='qz',
                           start=0,
                           stop=11,
                           num=4,
                           unit=sc.Unit('1/angstrom'))
        with file_location("test2.txt") as file_path:
            p.write_reflectometry(file_path, bins)
            written_data = np.loadtxt(file_path, unpack=True)
            assert_almost_equal(written_data[0],
                                bins.values[:-1] + np.diff(bins.values))
            assert_almost_equal(written_data[1], np.array([3, 3, 3]) / 9)
            assert_almost_equal(written_data[2], np.sqrt(np.array([3, 3, 3]) / 81))
            assert_almost_equal(written_data[3], np.linspace(0.325, 1.0, 3))

    def test_write_wavelength_theta(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(dims=["event"],
                                                   values=DETECTORS.astype(float),
                                                   unit=sc.units.angstrom)
        p.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        bins1 = sc.linspace(dim='wavelength',
                            start=0,
                            stop=100,
                            num=10,
                            unit=sc.units.angstrom)
        bins2 = sc.linspace(dim='theta', start=0, stop=100, num=10, unit=sc.units.deg)
        with file_location("test1.txt") as file_path:
            p.write_wavelength_theta(file_path, (bins1, bins2))
            written_data = np.loadtxt(file_path, unpack=True)
            assert_equal(written_data.shape, (11, 9))
