# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# flake8: noqa: E501
"""
Tests for write module
"""

# author: Andrew R. McCluskey (arm61)

import os
import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import scipp as sc
from ess.reflectometry import write, data, orso
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


class TestWrite(unittest.TestCase):
    def test_write(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["qz"] = sc.Variable(
            dims=["event"],
            values=np.linspace(1, 10, N),
            unit=sc.Unit('1/angstrom'),
        )
        p.event.coords["sigma_qz_by_qz"] = sc.Variable(
            dims=["event"],
            values=np.linspace(0.1, 1.0, N),
            unit=sc.Unit('1/angstrom'),
            dtype=sc.dtype.float64,
        )
        p.event.coords["tof"] = sc.Variable(dims=["event"],
                                            values=DETECTORS.astype(float))
        file_path = (os.path.dirname(os.path.realpath(__file__)) +
                     os.path.sep + "test1.txt")
        write.reflectometry(p, file_path)
        written_data = np.loadtxt(file_path, unpack=True)
        assert_equal(written_data.shape, (4, 199))

    def test_write_bins(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["qz"] = sc.Variable(
            dims=["event"],
            values=np.linspace(1, 10, N),
            unit=sc.Unit('1/angstrom'),
        )
        p.event.coords["sigma_qz_by_qz"] = sc.Variable(
            dims=["event"],
            values=np.linspace(0.1, 1.0, N),
            unit=sc.Unit('1/angstrom'),
            dtype=sc.dtype.float64,
        )
        p.event.coords["tof"] = sc.Variable(dims=["event"],
                                            values=DETECTORS.astype(float))
        bins = np.linspace(0, 11, 4)
        file_path = (os.path.dirname(os.path.realpath(__file__)) +
                     os.path.sep + "test2.txt")
        write.reflectometry(p, file_path, {"bins": bins}, 'hello')
        written_data = np.loadtxt(file_path, unpack=True)
        assert_almost_equal(written_data[0], bins[:-1] + np.diff(bins))
        assert_almost_equal(written_data[1], np.array([3, 3, 3]) / 9)
        assert_almost_equal(written_data[2], np.sqrt(np.array([3, 3, 3]) / 81))
        assert_almost_equal(written_data[3], np.linspace(0.325, 1.0, 3))
        f = open(file_path, 'r')
        assert_equal(f.readline(), '# hello\n')
        f.close()

    def test_write_wavelength_theta(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS.astype(float),
            unit=sc.units.angstrom)
        p.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        file_path = (os.path.dirname(os.path.realpath(__file__)) +
                     os.path.sep + "test1.txt")
        bins = np.linspace(0, 100, 10)
        write.wavelength_theta(p, file_path, (bins, bins))
        written_data = np.loadtxt(file_path, unpack=True)
        assert_equal(written_data.shape, (11, 9))

    def test_write_wavelength_theta_norm(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS.astype(float),
            unit=sc.units.angstrom)
        p.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        q = data.ReflData(BINNED.copy())
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
        write.wavelength_theta(z, file_path, (bins, bins), z.sample.orso)
        written_data = np.loadtxt(file_path, unpack=True)
        assert_equal(written_data.shape, (11, 9))

    def test_write_wavelength_theta_header(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS.astype(float),
            unit=sc.units.angstrom)
        p.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        file_path = (os.path.dirname(os.path.realpath(__file__)) +
                     os.path.sep + "test1.txt")
        bins = np.linspace(0, 100, 10)
        write.wavelength_theta(
            p, file_path, (bins, bins),
            orso.Orso(orso.Creator(), orso.DataSource(), orso.Reduction(), []))
        written_data = np.loadtxt(file_path, unpack=True)
        assert_equal(written_data.shape, (11, 9))
        f = open(file_path, 'r')
        assert_equal(
            f.readline(),
            '# # ORSO reflectivity data file | 0.1 standard | YAML encoding | https://reflectometry.org\n'
        )
        f.close()
