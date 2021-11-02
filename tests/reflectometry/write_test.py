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
                                   dtype=sc.dtype.int32),
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
BINNED.bins.constituents['data'].coords["tof"] = sc.linspace(dim="event",
                                                             start=1,
                                                             stop=10,
                                                             num=N,
                                                             unit=sc.units.us)
BINNED.attrs['sample_position'] = sc.geometry.position(0. * sc.units.m, 0. * sc.units.m,
                                                       0. * sc.units.m)
BINNED.attrs['instrument_name'] = sc.scalar(value='AMOR')
BINNED.attrs['experiment_title'] = sc.scalar(value='test')


class TestWrite(unittest.TestCase):
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
        p.event.coords["tof"] = sc.Variable(dims=["event"],
                                            values=DETECTORS.astype(float))
        bins = sc.linspace(dim='qz',
                           start=0,
                           stop=11,
                           num=4,
                           unit=sc.Unit('1/angstrom'))
        with file_location("test2.txt") as file_path:
            write.reflectometry(p, file_path, bins, 'hello')
            written_data = np.loadtxt(file_path, unpack=True)
            assert_almost_equal(written_data[0],
                                bins.values[:-1] + np.diff(bins.values))
            assert_almost_equal(written_data[1], np.array([3, 3, 3]) / 9)
            assert_almost_equal(written_data[2], np.sqrt(np.array([3, 3, 3]) / 81))
            assert_almost_equal(written_data[3], np.linspace(0.325, 1.0, 3))
            f = open(file_path, 'r')
            assert_equal(f.readline(), '# hello\n')
            f.close()

    def test_write_wavelength_theta(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(dims=["event"],
                                                   values=DETECTORS.astype(float),
                                                   unit=sc.units.angstrom)
        p.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        with file_location("test1.txt") as file_path:
            bins1 = sc.linspace(dim='wavelength',
                                start=0,
                                stop=100,
                                num=10,
                                unit=sc.Unit('angstrom'))
            bins2 = sc.linspace(dim='theta',
                                start=0,
                                stop=100,
                                num=10,
                                unit=sc.Unit('deg'))
            write.wavelength_theta(p, file_path, (bins1, bins2))
            written_data = np.loadtxt(file_path, unpack=True)
            assert_equal(written_data.shape, (11, 9))

    def test_write_wavelength_theta_norm(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(dims=["event"],
                                                   values=DETECTORS.astype(float),
                                                   unit=sc.units.angstrom)
        p.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        q = data.ReflData(BINNED.copy())
        q.event.coords["wavelength"] = sc.Variable(dims=["event"],
                                                   values=DETECTORS.astype(float),
                                                   unit=sc.units.angstrom)
        q.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        z = amor_data.Normalisation(p, q)
        with file_location("test1.txt") as file_path:
            bins1 = sc.linspace(dim='wavelength',
                                start=0,
                                stop=100,
                                num=10,
                                unit=sc.Unit('angstrom'))
            bins2 = sc.linspace(dim='theta',
                                start=0,
                                stop=100,
                                num=10,
                                unit=sc.Unit('deg'))
            write.wavelength_theta(z, file_path, (bins1, bins2), z.sample.orso)
            written_data = np.loadtxt(file_path, unpack=True)
            assert_equal(written_data.shape, (11, 9))

    def test_write_wavelength_theta_header(self):
        p = data.ReflData(BINNED.copy())
        p.event.coords["wavelength"] = sc.Variable(dims=["event"],
                                                   values=DETECTORS.astype(float),
                                                   unit=sc.units.angstrom)
        p.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS.astype(float),
                                              unit=sc.units.deg)
        with file_location("test1.txt") as file_path:
            bins1 = sc.linspace(dim='wavelength',
                                start=0,
                                stop=100,
                                num=10,
                                unit=sc.Unit('angstrom'))
            bins2 = sc.linspace(dim='theta',
                                start=0,
                                stop=100,
                                num=10,
                                unit=sc.Unit('deg'))
            write.wavelength_theta(
                p, file_path, (bins1, bins2),
                orso.Orso(orso.Creator(), orso.DataSource(), orso.Reduction(), []))
            written_data = np.loadtxt(file_path, unpack=True)
            assert_equal(written_data.shape, (11, 9))
            f = open(file_path, 'r')
            assert_equal(
                f.readline(),
                '# # ORSO reflectivity data file | 0.1 standard | YAML encoding | https://reflectometry.org\n'
            )
            f.close()
