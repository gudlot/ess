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
from ess.reflectometry import binning, data

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
BINNED.bins.constituents['data'].coords["tof"] = sc.linspace(dim='event',
                                                             start=1,
                                                             stop=10,
                                                             num=N,
                                                             unit=sc.units.us)
BINNED.bins.constituents['data'].coords["qz"] = sc.linspace(dim='event',
                                                            start=0.1,
                                                            stop=1.0,
                                                            num=N,
                                                            unit=sc.Unit('1/angstrom'))
BINNED.attrs['sample_position'] = sc.geometry.position(0. * sc.units.m, 0. * sc.units.m,
                                                       0. * sc.units.m)
BINNED.attrs['instrument_name'] = sc.scalar(value='AMOR')
BINNED.attrs['experiment_title'] = sc.scalar(value='test')


class TestBinning(unittest.TestCase):
    def test_q_bin(self):
        p = data.ReflData(BINNED.copy())
        bins = sc.linspace(dim='qz',
                           start=0.01,
                           stop=2,
                           num=10,
                           unit=sc.Unit('1/angstrom'))
        result = binning.q_bin(p.data, bins)
        assert_equal(result.shape, [9])
        assert_almost_equal(sc.max(result.coords['qz']).value, 2)

    def test_q_bin_unit(self):
        p = data.ReflData(BINNED.copy())
        bins = sc.linspace(dim='qz',
                           start=0.01e10,
                           stop=2e10,
                           num=5,
                           unit=sc.Unit('1/m'))
        result = binning.q_bin(p.data, bins)
        assert_equal(result.shape, [4])
        assert_almost_equal(sc.max(result.coords['qz']).value, 2e10)

    def test_2d_bin(self):
        p = data.ReflData(BINNED.copy())
        bins = (sc.linspace(dim='tof',
                            start=0.1e-6,
                            stop=10e-6,
                            num=50,
                            unit=sc.units.us),
                sc.linspace(dim='qz',
                            start=0.01e10,
                            stop=2e10,
                            num=50,
                            unit=sc.Unit('1/angstrom')))
        result = binning.two_dimensional_bin(p.data, bins)
        assert_equal(result.shape, [49, 49])

    def test_2d_bin_unit(self):
        p = data.ReflData(BINNED.copy())
        bins = (sc.linspace(dim='tof', start=0.1e-6, stop=10e-6, num=5,
                            unit=sc.units.s),
                sc.linspace(dim='qz',
                            start=0.01e10,
                            stop=2e10,
                            num=5,
                            unit=(1 / sc.units.m).unit))
        result = binning.two_dimensional_bin(p.data, bins)
        assert_equal(result.shape, [4, 4])
        assert_almost_equal(sc.max(result.coords['qz']).value, 2e10)
        assert_almost_equal(sc.max(result.coords['tof']).value, 10e-6)

    def test_2d_bin_bad_dim(self):
        p = data.ReflData(BINNED.copy())
        bins = (sc.linspace(dim='tof', start=0.1e-6, stop=10e-6, num=5,
                            unit=sc.units.s),
                sc.linspace(dim='q',
                            start=0.01e10,
                            stop=2e10,
                            num=5,
                            unit=(1 / sc.units.m).unit))
        with self.assertRaises(sc.NotFoundError):
            result = binning.two_dimensional_bin(p.data, bins)
