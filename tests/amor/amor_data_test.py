# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
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
from ..tools.io import file_location

np.random.seed(1)

# Make 20 neutrons in 2 pulses, so tof values extend up to 1.5e5 microseconds.
# ArmorData folds the 2 pulses and performs a tof correction.
N = 20
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
        "tof": sc.linspace(dim="event",
                           start=1.0e4,
                           stop=1.4e5,
                           num=N,
                           unit=sc.units.us)
    },
)

DETECTOR_ID = sc.Variable(dims=["detector_id"],
                          values=np.arange(1, 5),
                          dtype=sc.dtype.int32)
BINNED = sc.bin(DATA,
                edges=[sc.array(dims=["tof"], values=[0.5e4, 1.5e5], unit="us")],
                groups=[DETECTOR_ID])

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
BINNED.attrs['sample_position'] = sc.geometry.position(0. * sc.units.m, 0. * sc.units.m,
                                                       0. * sc.units.m)
BINNED.attrs['instrument_name'] = sc.scalar(value='AMOR')
BINNED.attrs['experiment_title'] = sc.scalar(value='test')


class TestAmorData(unittest.TestCase):
    def test_amordata_init(self):
        """
        Testing the default initialisation of the AmorData objects.
        """
        p = amor_data.AmorData(BINNED.copy())
        assert isinstance(p.data, sc._scipp.core.DataArray)
        assert isinstance(p.data.data, sc._scipp.core.Variable)
        # We cut off the wavelength at 2.4 angsrtoms, which discards 3 neutrons
        assert len(p.data.bins.constituents["data"].coords["wavelength"].values) == 17
        assert sc.allclose(
            sc.sort(p.data.bins.constituents["data"].coords["wavelength"], 'event'),
            sc.array(dims=['event'],
                     values=[
                         2.9689534, 3.2561593, 4.5305458, 4.8116494, 6.2246763,
                         6.4293591, 7.4802203, 8.1092884, 9.2674094, 9.3270709,
                         10.4877315, 10.8782681, 12.369804, 12.4294654, 13.4952427,
                         14.5801273, 16.1356174
                     ],
                     unit='angstrom'))
        assert sc.identical(
            p.data.meta['source_position'],
            sc.geometry.position(0. * sc.units.m, 0. * sc.units.m, -15. * sc.units.m))
        assert sc.allclose(p.data.coords['sigma_lambda_by_lambda'],
                           sc.array(dims=['detector_id'], values=[0.0130052] * 4))
        assert sc.identical(p.tau, sc.scalar(7.5e4, unit='us'))
        assert sc.identical(p.chopper_detector_distance, sc.scalar(19., unit='m'))
        assert sc.identical(p.chopper_chopper_distance, sc.scalar(0.49, unit='m'))
        assert sc.identical(p.chopper_phase, sc.scalar(-8.0, unit='deg'))
        assert sc.identical(p.wavelength_cut, sc.scalar(2.4, unit='angstrom'))

    def test_amordata_init_nodefault(self):
        """
        Testing the initialisation of the AmorData objects without defaults.
        """
        kwargs = {
            'reduction_creator': 'andrew',
            'data_owner': 'andrew',
            'reduction_creator_affiliation': 'ess',
            'experiment_id': '1234',
            'experiment_date': '2021-04-21',
            'sample_description': 'my sample',
            'reduction_file': 'my_notebook.ipynb',
            'chopper_phase': -5 * sc.units.deg,
            'chopper_chopper_distance': 0.3 * sc.units.m,
            'chopper_detector_distance': 18. * sc.units.m,
            'wavelength_cut': 1.4 * sc.units.angstrom
        }

        p = amor_data.AmorData(BINNED.copy(), **kwargs)
        # We cut off the wavelength at 1.4 angsrtoms, which discards 1 neutron
        assert len(p.data.bins.constituents["data"].coords["wavelength"].values) == 19
        assert sc.allclose(
            sc.sort(p.data.bins.constituents["data"].coords["wavelength"], 'event'),
            sc.array(dims=['event'],
                     values=[
                         1.79481481, 1.85447624, 3.24367799, 3.55170242, 4.80527036,
                         5.10719253, 6.5080681, 6.72490224, 7.75494484, 8.40483156,
                         9.55080124, 9.61046267, 10.76245608, 11.16165996, 12.65319581,
                         12.71285725, 13.76996732, 14.87567042, 16.43116053
                     ],
                     unit='angstrom'))
        assert sc.identical(
            p.data.meta['source_position'],
            sc.geometry.position(0. * sc.units.m, 0. * sc.units.m, -15. * sc.units.m))
        assert sc.allclose(p.data.coords['sigma_lambda_by_lambda'],
                           sc.array(dims=['detector_id'], values=[0.0079624] * 4))
        assert sc.identical(p.tau, sc.scalar(7.5e4, unit='us'))
        assert sc.identical(p.chopper_detector_distance,
                            kwargs['chopper_detector_distance'])
        assert sc.identical(p.chopper_chopper_distance,
                            kwargs['chopper_chopper_distance'])
        assert sc.identical(p.chopper_phase, kwargs['chopper_phase'])
        assert sc.identical(p.wavelength_cut, kwargs['wavelength_cut'])

        assert p.orso.creator.name == 'andrew'
        assert p.orso.creator.affiliation == 'ess'
        assert p.orso.data_source.owner == 'andrew'
        assert p.orso.data_source.experiment_id == '1234'
        assert p.orso.data_source.experiment_date == '2021-04-21'
        assert p.orso.reduction.script == 'my_notebook.ipynb'

    def test_wavelength_masking(self):
        p = amor_data.AmorData(BINNED.copy())
        p.wavelength_masking(
            wavelength_min=2 * sc.units.angstrom,
            wavelength_max=4 * sc.units.angstrom,
        )
        assert_equal(
            p.event.masks["wavelength"].values,
            ~((p.event.coords["wavelength"].values >= 2) &
              (p.event.coords["wavelength"].values <= 4)),
        )

    def test_wavelength_masking_no_min(self):
        p = amor_data.AmorData(BINNED.copy())
        p.wavelength_masking(wavelength_max=4 * sc.units.angstrom)
        assert_equal(
            p.event.masks["wavelength"].values,
            ~((p.event.coords["wavelength"].values >= 2.4) &
              (p.event.coords["wavelength"].values <= 4)),
        )

    def test_wavelength_masking_no_max(self):
        p = amor_data.AmorData(BINNED.copy())
        p.wavelength_masking(wavelength_min=2 * sc.units.angstrom)
        assert_equal(
            p.event.masks["wavelength"].values,
            ~((p.event.coords["wavelength"].values >= 2) &
              (p.event.coords["wavelength"].values <= 16.2)),
        )


class TestAmorReference(unittest.TestCase):
    def test_amorreference_init(self):
        """
        Testing the default initialisation of the AmorReference objects.
        """
        p = amor_data.AmorReference(BINNED.copy())
        assert_equal(isinstance(p.data, sc._scipp.core.DataArray), True)
        assert_equal(isinstance(p.data.data, sc._scipp.core.Variable), True)
        assert_equal(p.m_value.value, 5)
        assert_almost_equal(p.event.coords['normalisation'].values, np.ones(17))

    def test_amorreference_init_nodefault(self):
        """
        Testing the default initialisation of the AmorReference objects.
        """
        p = amor_data.AmorReference(BINNED.copy(), m_value=4 * sc.units.dimensionless)
        assert_equal(isinstance(p.data, sc._scipp.core.DataArray), True)
        assert_equal(isinstance(p.data.data, sc._scipp.core.Variable), True)
        assert_equal(p.m_value.value, 4)
        assert_almost_equal(p.event.coords['normalisation'].values, np.ones(17))


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
        nevents = p.event.coords["wavelength"].sizes['event']
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS[:nevents].astype(float),
            unit=sc.units.angstrom)
        p.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS[:nevents].astype(float),
                                              unit=sc.units.deg)
        q = amor_data.AmorData(BINNED.copy())
        q.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS[:nevents].astype(float),
            unit=sc.units.angstrom)
        q.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS[:nevents].astype(float),
                                              unit=sc.units.deg)
        z = amor_data.Normalisation(p, q)
        del z.sample.event.coords['qz']
        with self.assertRaises(sc.NotFoundError):
            z.q_bin()

    def test_write_reflectometry_norm(self):
        p = amor_data.AmorData(BINNED.copy())
        nevents = p.event.coords["wavelength"].sizes['event']
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS[:nevents].astype(float),
            unit=sc.units.angstrom)
        p.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS[:nevents].astype(float),
                                              unit=sc.units.deg)
        q = amor_data.AmorData(BINNED.copy())
        q.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS[:nevents].astype(float),
            unit=sc.units.angstrom)
        q.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS[:nevents].astype(float),
                                              unit=sc.units.deg)
        z = amor_data.Normalisation(p, q)
        with file_location("test1.txt") as file_path:
            z.write_reflectometry(file_path)
            written_data = np.loadtxt(file_path, unpack=True)
            assert_equal(written_data.shape, (4, 199))

    def test_binwavelength_theta_norm(self):
        p = amor_data.AmorData(BINNED.copy())
        nevents = p.event.coords["wavelength"].sizes['event']
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS[:nevents].astype(float),
            unit=sc.units.angstrom)
        p.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS[:nevents].astype(float),
                                              unit=sc.units.deg)
        q = amor_data.AmorData(BINNED.copy())
        q.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS[:nevents].astype(float),
            unit=sc.units.angstrom)
        q.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS[:nevents].astype(float),
                                              unit=sc.units.deg)
        z = amor_data.Normalisation(p, q)
        bins1 = sc.linspace(dim='wavelength',
                            start=0,
                            stop=100,
                            num=10,
                            unit=sc.units.angstrom)
        bins2 = sc.linspace(dim='theta', start=0, stop=100, num=10, unit=sc.units.deg)
        k = z.wavelength_theta_bin((bins1, bins2))
        assert_equal(k.shape, (9, 9))

    def test_write_wavelength_theta_norm(self):
        p = amor_data.AmorData(BINNED.copy())
        nevents = p.event.coords["wavelength"].sizes['event']
        p.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS[:nevents].astype(float),
            unit=sc.units.angstrom)
        p.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS[:nevents].astype(float),
                                              unit=sc.units.deg)
        q = amor_data.AmorData(BINNED.copy())
        q.event.coords["wavelength"] = sc.Variable(
            dims=["event"],
            values=DETECTORS[:nevents].astype(float),
            unit=sc.units.angstrom)
        q.event.coords["theta"] = sc.Variable(dims=["event"],
                                              values=DETECTORS[:nevents].astype(float),
                                              unit=sc.units.deg)
        z = amor_data.Normalisation(p, q)
        with file_location("test1.txt") as file_path:
            bins1 = sc.linspace(dim='wavelength',
                                start=0,
                                stop=100,
                                num=10,
                                unit=sc.units.angstrom)
            bins2 = sc.linspace(dim='theta',
                                start=0,
                                stop=100,
                                num=10,
                                unit=sc.units.deg)
            z.write_wavelength_theta(file_path, (bins1, bins2))
            written_data = np.loadtxt(file_path, unpack=True)
            assert_equal(written_data.shape, (11, 9))
