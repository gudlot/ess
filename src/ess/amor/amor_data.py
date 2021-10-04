# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# flake8: noqa: E501
"""
This is the class for data reduction from the Amor instrument, which is a subclass of the broader `ReflData` class.
Features of this class included correcting for the time-of-flight measurement at Amor.
"""
import copy
import scipp as sc
import scippneutron as scn
import numpy as np
from ess.reflectometry import HDM, orso, write
from ess.reflectometry.data import ReflData


class AmorData(ReflData):
    """
    Reduction of a single Amor dataset.

    Args:
        data (:py:class:`scipp._scipp.core.DataArray` or :py:attr:`str`): The data to be reduced or the path to the file to be reduced.
        reduction_creator (:py:attr:`str`, optional): The name of the creator of the reduction. Optional, default :code:`None`.
        data_owner (:py:attr:`str`, optional): The name of the owner of the data. Optional, default :code:`None`.
        experiment_id (:py:attr:`str`, optional): The experimental identifier. Optional, default :code:`None`.
        experiment_date (:py:attr:`str`, optional): The date or date range for the experiment. Optional, default :code:`None`.
        sample_description (:py:attr:`str`, optional): A short description of the sample. Optional, default :code:`None`.
        reduction_file (:py:attr:`str`, optional): The name of the file used for reduction (:code:`.py` script or :code:`.ipynb` notebook). Optional, default :code:`None`.
        data_file (:py:attr:`str`, optional): If a :py:class:`scipp._scipp.core.DataArray` is given as the :py:attr:`data`, a :py:attr:`data_file` should be defined for output in the file. Optional, default :code:`None`.
        reduction_creator_affiliation (:py:attr:`str`, optional): The affiliation of the reduction owner. Optional, defaults to :code:`None`.
        sample_angle_offset (:py:class:`scipp.Variable`, optional): Correction for omega or possibly misalignment of sample. Optional, default :code:`0 degrees of arc`.
        gravity (:py:attr:`bool`, optional): Should gravity be accounted for. Optional, default `True`.
        beam_size (:py:class:`scipp._scipp.core.Variable`, optional): Size of the beam perpendicular to the scattering surface. Optional, default :code:`0.001 m`.
        sample_size (:py:class:`scipp._scipp.core.Variable`, optional): Size of the sample in direction of the beam. Optional, default :code:`0.01 m`.
        detector_spatial_resolution (:py:class:`scipp._scipp.core.Variable`, optional): Spatial resolution of the detector. Optional, default :code:`2.5 mm`
        chopper_sample_distance (:py:class:`scipp._scipp.core.Variable`, optional): Distance from chopper to sample. Optional, default :code:`15. m`
        chopper_speed (:py:class:`scipp._scipp.core.Variable`, optional): Rotational velocity of the chopper. Optional, default :code:`6.6666... e-6 µs^{-1}`.
        chopper_detector_distance (:py:class:`scipp._scipp.core.Variable`, optional): Distance from chopper to detector. Optional, default :code:`19 m`.
        chopper_chopper_distance (:py:class:`scipp._scipp.core.Variable`, optional): The distance between the wavelength defining choppers. Optional, default :code:`0.49 m`
        chopper_phase (:py:class:`scipp._scipp.core.Variable`, optional): Phase offset between chopper pulse and ToF zero. Optional, default :code:`-8. degrees of arc`.
        wavelength_cut (:py:class:`scipp._scipp.core.Variable`, optional): Minimum cutoff for wavelength. Optional, default :code:`2.4 Å`.

    Attributes:
        tau (:py:class:`scipp._scipp.core.Variable`): Half of the inverse of the chopper speed.
    """
    def __init__(
        self,
        data,
        reduction_creator=None,
        data_owner=None,
        experiment_id=None,
        experiment_date=None,
        sample_description=None,
        reduction_file=None,
        data_file=None,
        reduction_creator_affiliation=None,
        sample_angle_offset=0 * sc.units.deg,
        gravity=True,
        beam_size=0.001 * sc.units.m,
        sample_size=0.01 * sc.units.m,
        detector_spatial_resolution=0.0025 * sc.units.m,
        chopper_sample_distance=15.0 * sc.units.m,
        chopper_speed=20 / 3 * 1e-6 / sc.units.us,
        chopper_detector_distance=19.0 * sc.units.m,
        chopper_chopper_distance=0.49 * sc.units.m,
        chopper_phase=-8.0 * sc.units.deg,
        wavelength_cut=2.4 * sc.units.angstrom,
    ):
        super().__init__(
            data,
            sample_angle_offset=sample_angle_offset,
            gravity=gravity,
            beam_size=beam_size,
            sample_size=sample_size,
            detector_spatial_resolution=detector_spatial_resolution,
            data_file=data_file,
        )
        # Convert tof nanoseconds to microseconds for convenience
        # TODO: is it safe to assume that the dtype of the binned wrapper coordinate is
        # the same as the dtype of the underlying event coordinate?
        if self.data.coords['tof'].dtype != sc.dtype.float64:
            self.data.bins.coords['tof'] = self.data.bins.coords['tof'].astype(
                'float64')
            self.data.coords['tof'] = self.data.coords['tof'].astype('float64')
        self.data.bins.coords['tof'] = sc.to_unit(self.data.bins.coords['tof'], 'us')
        self.data.coords['tof'] = sc.to_unit(self.data.coords['tof'], 'us')
        # These are Amor specific parameters
        self.tau = 1 / (2 * chopper_speed)
        self.chopper_detector_distance = chopper_detector_distance
        self.chopper_chopper_distance = chopper_chopper_distance
        self.chopper_phase = chopper_phase
        self.wavelength_cut = wavelength_cut
        # The source position is not the true source position due to the
        # use of choppers to define the pulse.
        self.data.coords["source_position"] = sc.geometry.position(
            0.0 * sc.units.m, 0.0 * sc.units.m, -chopper_sample_distance)
        self.tof_correction()
        # The wavelength contribution to the resolution function, defined
        # by the distance between the two choppers.
        # Division by 2np.sqrt(2*np.log(2)) converts from FWHM to std.
        self.data.coords["sigma_lambda_by_lambda"] = self.chopper_chopper_distance / (
            self.data.coords["position"].fields.z -
            self.data.coords["source_position"].fields.z)
        self.data.coords["sigma_lambda_by_lambda"] /= 2 * np.sqrt(2 * np.log(2))
        self.find_wavelength(wavelength_cut=self.wavelength_cut)
        self.find_theta()
        self.illumination()
        self.find_qz()
        self._setup_orso(reduction_creator, reduction_creator_affiliation,
                         sample_description, data_owner, experiment_id, experiment_date,
                         reduction_file)

    def _setup_orso(self, reduction_creator, reduction_creator_affiliation,
                    sample_description, data_owner, experiment_id, experiment_date,
                    reduction_file):
        """
        Setup the ORSO header object.

        Args:
            reduction_creator (:py:attr:`str`, optional): The name of the creator of the reduction. Optional, default :code:`None`.
            reduction_creator_affiliation (:py:attr:`str`, optional): The affiliation of the reduction owner. Optional, defaults to :code:`None`.
            data_owner (:py:attr:`str`, optional): The name of the owner of the data. Optional, default :code:`None`.
            experiment_id (:py:attr:`str`, optional): The experimental identifier. Optional, default :code:`None`.
            experiment_date (:py:attr:`str`, optional): The date or date range for the experiment. Optional, default :code:`None`.
            sample_description (:py:attr:`str`, optional): A short description of the sample. Optional, default :code:`None`.
            reduction_file (:py:attr:`str`, optional): The name of the file used for reduction (:code:`.py` script or :code:`.ipynb` notebook). Optional, default :code:`None`.
        """
        measurement = orso.Measurement(
            'energy-dispersive',
            orso.ValueScalar(
                sc.mean(self.event.coords['theta']).value,
                str(self.event.coords['theta'].unit)),
            orso.ValueRange(
                sc.min(self.event.coords['wavelength']).value,
                sc.max(self.event.coords['wavelength']).value,
                str(self.event.coords['wavelength'].unit)))
        self.orso.data_source.measurement = measurement
        self.orso.data_source.facility = 'Paul Scherrer Institut, SINQ'
        self.orso.columns.append(orso.Column('Qz', '1/angstrom'))
        self.orso.columns.append(orso.Column('RQz', 'dimensionless'))
        self.orso.columns.append(orso.Column('sR', 'dimensionless'))
        self.orso.columns.append(
            orso.Column(
                'sQ/Qz', 'dimensionless',
                'fractional description of sigma from Gaussian resolution function'))
        if reduction_creator is not None:
            self.orso.creator.name = reduction_creator
        if reduction_creator_affiliation is not None:
            self.orso.creator.affiliation = reduction_creator_affiliation
        if sample_description is not None:
            self.orso.data_source.experiment.sample = orso.Sample(sample_description)
        if data_owner is not None:
            self.orso.data_source.owner = data_owner
        if experiment_id is not None:
            self.orso.data_source.experiment_id = experiment_id
        if experiment_date is not None:
            self.orso.data_source.experiment_date = experiment_date
        if reduction_file is not None:
            self.orso.reduction.script = reduction_file
        try:
            self.orso.reduction.input_files = orso.Files([orso.File(self.data_file)],
                                                         [None])
        except TypeError:
            self.orso.reduction.input_files = orso.Files([None], [None])
            self.orso.reduction.comment = 'Live reduction'

    def tof_correction(self):
        """
        A correction for the presense of the chopper with respect to the "true" ToF.
        Also fold the two pulses.
        TODO: generalise mechanism to fold any number of pulses.
        """
        # Make 2 bins, one for each pulse
        edges = sc.array(dims=['tof'],
                         values=[0., self.tau.value, 2 * self.tau.value],
                         unit=self.tau.unit)
        self.data = sc.bin(self.data, edges=[edges])
        # Make one offset for each bin
        tof_offset = self.tau * self.chopper_phase / (180.0 * sc.units.deg)
        offset = sc.concatenate(tof_offset, tof_offset - self.tau, 'tof')
        # Apply the offset on both bins
        self.data.bins.coords['tof'] += offset
        # Rebin to exclude second (empty) pulse range
        self.data = sc.bin(self.data,
                           edges=[sc.concatenate(0. * sc.units.us, self.tau, 'tof')])

    def wavelength_masking(self, wavelength_min=None, wavelength_max=None):
        """
        Overwriting the :py:class:`ess.reflectometry.data.ReflData` wavelength masking functionality.

        Args:
            wavelength_min (:py:class:`scipp._scipp.core.Variable`, optional): Minimum wavelength to be used. Optional, default to :code:`wavelength_cut` value.
            wavelength_max (:py:class:`scipp._scipp.core.Variable`, optional): Maximum wavelength to be used. Optional, default to :code:`wavelength_min + tau * (ess.reflectometry.HDM / chopper_detector_distance)`.
        """
        if wavelength_min is None:
            wavelength_min = self.wavelength_cut
        if wavelength_max is None:
            wavelength_max = wavelength_min + sc.to_unit(
                self.tau * (HDM / self.chopper_detector_distance), 'angstrom')
            if (wavelength_max > sc.max(self.event.coords["wavelength"])).value:
                wavelength_max = sc.max(self.event.coords["wavelength"])
        wavelength_max = sc.to_unit(wavelength_max,
                                    self.event.coords['wavelength'].unit)
        wavelength_min = sc.to_unit(wavelength_min,
                                    self.event.coords['wavelength'].unit)
        self.data.bins.masks["wavelength"] = (
            self.data.bins.coords["wavelength"] <
            wavelength_min) | (self.data.bins.coords["wavelength"] > wavelength_max)


class AmorReference(AmorData):
    """
    Additional functionality over the :py:class:`ess.amor.AmorData` class for use with reference supermirror measurements.

    Args:
        data (:py:class:`scipp._scipp.core.DataArray` or :py:attr:`str`): The data to be reduced or the path to the file to be reduced.
        sample_angle_offset (:py:class:`scipp.Variable`, optional): Correction for omega or possibly misalignment of sample. Optional, default :code:`0 degrees of arc`.
        gravity (:py:attr:`bool`, optional): Should gravity be accounted for. Optional, default :code:`True`.
        beam_size (:py:class:`scipp._scipp.core.Variable`, optional): Size of the beam perpendicular to the scattering surface. Optional, default :code:`0.001 m`.
        sample_size (:py:class:`scipp._scipp.core.Variable`, optional): Size of the sample in direction of the beam. Optional, default :code:`0.01 m`.
        detector_spatial_resolution (:py:class:`scipp._scipp.core.Variable`, optional): Spatial resolution of the detector. Optional, default :code:`2.5 mm`
        chopper_sample_distance (:py:class:`scipp._scipp.core.Variable`, optional): Distance from chopper to sample. Optional, default :code:`15. m,`
        chopper_speed (:py:class:`scipp._scipp.core.Variable`, optional): Rotational velocity of the chopper. Optional, default :code:`6.6666... e-6 µs^{-1}`.
        chopper_detector_distance (:py:class:`scipp._scipp.core.Variable`, optional): Distance from chopper to detector. Optional, default :code:`19 m`.
        chopper_chopper_distance (:py:class:`scipp._scipp.core.Variable`, optional): The distance between the wavelength defining choppers. Optional, default :code:`0.49 m`
        chopper_phase (:py:class:`scipp._scipp.core.Variable`, optional): Phase offset between chopper pulse and ToF zero. Optional, default :code:`-8. degrees of arc`.
        wavelength_cut (:py:class:`scipp._scipp.core.Variable`, optional): Minimum cutoff for wavelength. Optional, default :code:`2.4 Å`.
        m_value (:py:class:`scipp._scipp.core.Variable`, optional): m-value of supermirror for reference. Optional, default :code:`5`.
        data_file (:py:attr:`str`, optional): If a :py:class:`scipp._scipp.core.DataArray` is given as the :py:attr:`data`, a :py:attr:`data_file` should be defined for output in the file. Optional, default :code:`None`.
        supermirror_critical_edge (:py:class:`scipp._scipp.core.Variable`, optional): The q-value at the critial edge for the supermirror. Optional, defaults to :code:`0.022 Å``.
        supermirror_alpha (:py:attr:`float`): The alpha value for the supermirror. Optional, defaults to :code:`2.841 Å`.

    Attributes:
        tau (:py:class:`scipp._scipp.core.Variable`): Half of the inverse of the chopper speed.
    """
    def __init__(self,
                 data,
                 sample_angle_offset=0 * sc.units.deg,
                 gravity=True,
                 beam_size=0.001 * sc.units.m,
                 sample_size=0.01 * sc.units.m,
                 detector_spatial_resolution=0.0025 * sc.units.m,
                 chopper_sample_distance=15.0 * sc.units.m,
                 chopper_speed=20 / 3 * 1e-6 / sc.units.us,
                 chopper_detector_distance=1.9e11 * sc.units.angstrom,
                 chopper_chopper_distance=0.49 * sc.units.m,
                 chopper_phase=-8.0 * sc.units.deg,
                 wavelength_cut=2.4 * sc.units.angstrom,
                 m_value=5 * sc.units.dimensionless,
                 data_file=None,
                 supermirror_critical_edge=((0.022) * sc.Unit('1/angstrom')),
                 supermirror_alpha=0.25 / 0.088):
        super().__init__(
            data,
            sample_angle_offset=sample_angle_offset,
            gravity=gravity,
            beam_size=beam_size,
            sample_size=sample_size,
            detector_spatial_resolution=detector_spatial_resolution,
            chopper_sample_distance=chopper_sample_distance,
            chopper_speed=chopper_speed,
            chopper_detector_distance=chopper_detector_distance,
            chopper_chopper_distance=chopper_chopper_distance,
            chopper_phase=chopper_phase,
            wavelength_cut=wavelength_cut,
            data_file=data_file,
        )
        self.m_value = m_value
        # The normalisation between the min and max of the supermirror is
        # normalised based on the characteristic of the supermirror at Amor
        supermirror_max_q = self.m_value * supermirror_critical_edge
        self.event.coords['normalisation'] = sc.ones(dims=['event'],
                                                     shape=self.event.data.shape)
        self.event.masks['normalisation'] = self.event.coords['qz'] >= supermirror_max_q
        self.event.coords['normalisation'].values[
            self.event.coords['qz'].values <
            supermirror_max_q.value] = self.event.coords['normalisation'].values[
                self.event.coords['qz'].values < supermirror_max_q.value] / (
                    1.0 - (supermirror_alpha) * (self.event.coords['qz'].values[
                        self.event.coords['qz'].values < supermirror_max_q.value] -
                                                 supermirror_critical_edge.value))
        self.event.coords['normalisation'].values[
            self.event.coords['qz'].values < supermirror_critical_edge.value] = 1
        self.data.bins.constituents['data'].data = self.event.data / self.event.coords[
            'normalisation'].astype(sc.dtype.float32)


class Normalisation:
    """
    Normalisation between a sample and a reference measurement.

    Args:
        sample (:py:class:`ess.reflectometry.ReflData` or :py:class:`ess.reflectometry.AmorData`): The sample to be normalised.
        reference (:py:class:`ess.reflectometry.AmorReference`): The reference measurement to normalise to.
    """
    def __init__(self, sample, reference):
        self.sample = sample
        self.reference = reference
        if self.reference.data_file is None:
            self.sample.orso.reduction.comment = 'Live reduction'
        else:
            self.sample.orso.reduction.input_files.reference_files = [
                orso.File(self.reference.data_file)
            ]

    def q_bin(self, bins=None, unit=sc.Unit('1/angstrom')):
        """
        Perform q-binned normalisation.

        Args:
            bins (:py:attr:`array_like`, optional): q-bin edges. Optional, defaults to the minimum q-range available with 100 bins.
            unit (:py:class:`scipp._scipp.core.Unit`, optional): Unit for q. Optional, defaults to 1/Å.

        Returns:
            (:py:class:`scipp._scipp.core.DataArray`): Normalised data array binned into qz with resolution.
        """
        if "qz" in self.sample.event.coords and "qz" in self.reference.event.coords:
            if bins is None:
                self.sample.event.coords['qz'] = sc.to_unit(
                    self.sample.event.coords['qz'], unit)
                self.reference.event.coords['qz'] = sc.to_unit(
                    self.reference.event.coords['qz'], unit)
                min_q = np.max([
                    sc.min(self.sample.event.coords['qz']).value,
                    sc.min(self.reference.event.coords['qz']).value
                ]) * unit
                max_q = np.min([
                    sc.max(self.sample.event.coords['qz']).value,
                    sc.max(self.reference.event.coords['qz']).value
                ]) * unit
                bins = sc.linspace(dim='qz',
                                   start=min_q.value,
                                   stop=max_q.value,
                                   num=200,
                                   unit=sc.Unit('1/angstrom'))
            binned_sample = self.sample.q_bin(bins)
            binned_reference = self.reference.q_bin(bins)
            del binned_reference.coords['sigma_qz_by_qz']
        else:
            raise sc.NotFoundError("qz coordinate cannot be found.")
        return binned_sample / binned_reference

    def wavelength_theta_bin(self, bins):
        """
        Perform wavelength/theta-binned normalisation.

        Args:
            bins (:py:attr:`tuple` of :py:attr:`array_like`): wavelength and theta edges.

        Returns:
            (:py:class:`scipp._scipp.core.DataArray`): Normalised data array binned into wavelength and theta.
        """
        return self.sample.wavelength_theta_bin(
            bins).bins.sum() / self.reference.wavelength_theta_bin(bins).bins.sum()

    def write_reflectometry(self, filename, bin_kwargs=None):
        """
        Write the reflectometry intensity data to a file.

        Args:
            filename (:py:attr:`str`): The file path for the file to be saved to.
            bin_kwargs (:py:attr:`dict`, optional): A dictionary of keyword arguments to be passed to the :py:func:`q_bin` class method. Optional, default is that default :py:func:`q_bin` keywords arguments are used.
        """
        write.reflectometry(self, filename, bin_kwargs, self.sample.orso)

    def write_wavelength_theta(self, filename, bins):
        """
        Write the reflectometry intensity data as a function of wavelength-theta to a file.

        Args:
            filename (:py:attr:`str`): The file path for the file to be saved to.
            bins (:py:attr:`tuple` of :py:attr:`array_like`): wavelength and theta edges.
        """
        write.wavelength_theta(self, filename, bins, self.sample.orso)
