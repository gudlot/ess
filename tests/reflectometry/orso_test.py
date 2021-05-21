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
from ess.reflectometry import orso
from ess import __version__ as VERSION
import getpass
import socket


class TestOrso(unittest.TestCase):
    def test_creator(self):
        c = orso.Creator()
        assert_equal(c.name, getpass.getuser())
        assert_equal(c.system, socket.gethostname())
        with self.assertRaises(AttributeError):
            _ = c.affiliation

    def test_creator_name(self):
        c = orso.Creator('andrew')
        assert_equal(c.name, 'andrew')
        assert_equal(c.system, socket.gethostname())
        with self.assertRaises(AttributeError):
            _ = c.affiliation

    def test_creator_name_affil(self):
        c = orso.Creator('andrew', 'ess')
        assert_equal(c.name, 'andrew')
        assert_equal(c.affiliation, 'ess')
        assert_equal(c.system, socket.gethostname())

    def test_creator_name_affil_time(self):
        c = orso.Creator('andrew', 'ess', '1992-07-14T00:00:00')
        assert_equal(c.name, 'andrew')
        assert_equal(c.affiliation, 'ess')
        assert_equal(c.time, '1992-07-14T00:00:00')
        assert_equal(c.system, socket.gethostname())

    def test_creator_name_affil_time_system(self):
        c = orso.Creator('andrew', 'ess', '1992-07-14T00:00:00', 'dmsc')
        assert_equal(c.name, 'andrew')
        assert_equal(c.affiliation, 'ess')
        assert_equal(c.time, '1992-07-14T00:00:00')
        assert_equal(c.system, 'dmsc')

    def test_sample(self):
        c = orso.Sample('nickel')
        assert_equal(c.name, 'nickel')

    def test_value_scalar(self):
        c = orso.ValueScalar(1., 'm')
        assert_almost_equal(c.magnitude, 1)
        assert_equal(c.unit, 'm')

    def test_value_scalar_repr(self):
        c = orso.ValueScalar(1., 'm')
        assert_equal(c.__repr__(), 'magnitude: 1.0\nunit: m\n')

    def test_value_scalar_angstrom(self):
        c = orso.ValueScalar(1., '\xC5')
        assert_almost_equal(c.magnitude, 1)
        assert_equal(c.unit, 'angstrom')

    def test_value_range(self):
        c = orso.ValueRange(1., 2., 'm')
        assert_almost_equal(c.min, 1)
        assert_almost_equal(c.max, 2)
        assert_equal(c.unit, 'm')

    def test_value_range_angstrom(self):
        c = orso.ValueRange(1., 2., '\xC5')
        assert_almost_equal(c.min, 1)
        assert_almost_equal(c.max, 2)
        assert_equal(c.unit, 'angstrom')

    def test_measurement(self):
        c = orso.Measurement('energy-dispersive', orso.ValueScalar(0.5, 'deg'),
                             orso.ValueRange(2.5, 10., 'angstrom'))
        assert_equal(c.scheme, 'energy-dispersive')
        assert_almost_equal(c.omega.magnitude, 0.5)
        assert_equal(c.omega.unit, 'deg')
        assert_almost_equal(c.wavelength.min, 2.5)
        assert_almost_equal(c.wavelength.max, 10.)
        assert_equal(c.wavelength.unit, 'angstrom')

    def test_measurement_polarisation(self):
        c = orso.Measurement('energy-dispersive',
                             orso.ValueScalar(0.5, 'deg'),
                             orso.ValueRange(2.5, 10., 'angstrom'),
                             polarisation='+')
        assert_equal(c.scheme, 'energy-dispersive')
        assert_almost_equal(c.omega.magnitude, 0.5)
        assert_equal(c.omega.unit, 'deg')
        assert_almost_equal(c.wavelength.min, 2.5)
        assert_almost_equal(c.wavelength.max, 10.)
        assert_equal(c.wavelength.unit, 'angstrom')
        assert_equal(c.polarisation, '+')

    def test_experiment(self):
        c = orso.Experiment('Amor', 'neutron', orso.Sample('nickel'))
        assert_equal(c.instrument, 'Amor')
        assert_equal(c.probe, 'neutron')
        assert_equal(c.sample.name, 'nickel')

    def test_datasource(self):
        c1 = orso.Measurement('energy-dispersive',
                              orso.ValueScalar(0.5, 'deg'),
                              orso.ValueRange(2.5, 10., 'angstrom'))
        c2 = orso.Experiment('Amor', 'neutron', orso.Sample('nickel'))
        c = orso.DataSource('andrew', 'ess', '123', '2021-04-20', 'experiment',
                            c2, c1)
        assert_equal(c.owner, 'andrew')
        assert_equal(c.facility, 'ess')
        assert_equal(c.experiment_id, '123')
        assert_equal(c.experiment_date, '2021-04-20')
        assert_equal(c.experiment.instrument, 'Amor')
        assert_equal(c.measurement.scheme, 'energy-dispersive')

    def test_file(self):
        c = orso.File(os.path.dirname(__file__) + os.sep + 'data_test.py')
        assert_equal(c.file,
                     os.path.dirname(__file__) + os.sep + 'data_test.py')

    def test_file_not_exist(self):
        with self.assertRaises(FileNotFoundError):
            c = orso.File(
                os.path.dirname(__file__) + os.sep + 'data_test_sadaoi.py')

    def test_files(self):
        f = orso.File(os.path.dirname(__file__) + os.sep + 'data_test.py')
        c = orso.Files([f])
        assert_equal(c.data_files[0].file,
                     os.path.dirname(__file__) + os.sep + 'data_test.py')
        with self.assertRaises(AttributeError):
            _ = c.reference_files

    def test_files_with_ref(self):
        f1 = orso.File(os.path.dirname(__file__) + os.sep + 'data_test.py')
        f2 = orso.File(
            os.path.dirname(__file__) + os.sep + 'corrections_test.py')
        c = orso.Files([f1], [f2])
        assert_equal(c.data_files[0].file,
                     os.path.dirname(__file__) + os.sep + 'data_test.py')
        assert_equal(
            c.reference_files[0].file,
            os.path.dirname(__file__) + os.sep + 'corrections_test.py')

    def test_reduction(self):
        f1 = orso.File(os.path.dirname(__file__) + os.sep + 'data_test.py')
        f2 = orso.File(
            os.path.dirname(__file__) + os.sep + 'corrections_test.py')
        f = orso.Files([f1], [f2])
        c = orso.Reduction('a_script.py', f)
        assert_equal(c.software, f'ess-{VERSION}')
        assert_equal(c.script, 'a_script.py')
        assert_equal(c.input_files.data_files[0].file,
                     os.path.dirname(__file__) + os.sep + 'data_test.py')
        assert_equal(
            c.input_files.reference_files[0].file,
            os.path.dirname(__file__) + os.sep + 'corrections_test.py')

    def test_reduction_comment(self):
        f1 = orso.File(os.path.dirname(__file__) + os.sep + 'data_test.py')
        f2 = orso.File(
            os.path.dirname(__file__) + os.sep + 'corrections_test.py')
        f = orso.Files([f1], [f2])
        c = orso.Reduction('a_script.py', f, 'hi')
        assert_equal(c.software, f'ess-{VERSION}')
        assert_equal(c.script, 'a_script.py')
        assert_equal(c.input_files.data_files[0].file,
                     os.path.dirname(__file__) + os.sep + 'data_test.py')
        assert_equal(
            c.input_files.reference_files[0].file,
            os.path.dirname(__file__) + os.sep + 'corrections_test.py')
        assert_equal(c.comment, 'hi')

    def test_reduction_noscript(self):
        f1 = orso.File(os.path.dirname(__file__) + os.sep + 'data_test.py')
        f2 = orso.File(
            os.path.dirname(__file__) + os.sep + 'corrections_test.py')
        f = orso.Files([f1], [f2])
        c = orso.Reduction(None, f, 'hi')
        assert_equal(c.software, f'ess-{VERSION}')
        assert_equal(c.input_files.data_files[0].file,
                     os.path.dirname(__file__) + os.sep + 'data_test.py')
        assert_equal(
            c.input_files.reference_files[0].file,
            os.path.dirname(__file__) + os.sep + 'corrections_test.py')
        assert_equal(c.comment, 'hi')

    def test_column(self):
        c = orso.Column('Qz', '1/angstrom')
        assert_equal(c.quantity, 'Qz')
        assert_equal(c.unit, '1/angstrom')

    def test_column_comment(self):
        c = orso.Column('Qz', '1/angstrom', 'hi')
        assert_equal(c.quantity, 'Qz')
        assert_equal(c.unit, '1/angstrom')
        assert_equal(c.comment, 'hi')
