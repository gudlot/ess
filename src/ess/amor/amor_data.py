# flake8: noqa: E501
"""
This is the class for data reduction from the Amor instrument, which is a subclass of the broader `ReflData` class.
Features of this class included correcting for the time-of-flight measurement at Amor.
"""
import scipp as sc
import scippneutron as scn
import numpy as np
from ess.reflectometry import HDM, orso
from ess.reflectometry.data import ReflData


class AmorData(ReflData):
    """
    Reduction of AMOR data.
    """
    def __init__(
        self,
        data,
        reduction_creator=None,
        data_owner=None,
        experiment_id=None,
        experiment_date=None,
        sample_description=None,
        notebook_file=None,
        data_file=None,
        reduction_creator_affiliation=None,
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
    ):
        """
        Args:
            data (`scipp._scipp.core.DataArray`): The data to be reduced.
            sample_angle_offset (`scipp.Variable`, optional): Correction for omega or possibly misalignment of sample. Optional, default `0 degrees of arc`.
            gravity (`bool`, optional): Should gravity be accounted for. Optional, default `True`.
            beam_size (`sc.Variable`, optional): Size of the beam perpendicular to the scattering surface. Optional, default `0.001 m`.
            sample_size (`sc.Variable`, optional): Size of the sample in direction of the beam. Optional, default `0.01 m`.
            detector_spatial_resolution (`sc.Variable`, optional): Spatial resolution of the detector. Optional, default `2.5 mm`
            chopper_sample_distance (`sc.Variable`, optional): Distance from chopper to sample. Optional, default `15.*sc.units.m,`
            chopper_speed (`sc.Variable`, optional): Rotational velocity of the chopper. Optional, default `6.6666... e-6 µs^{-1}`.
            chopper_detector_distance (`sc.Variable`, optional): Distance from chopper to detector. Optional, default `19 m`.
            chopper_chopper_distance (`sc.Variable`, optional): The distance between the wavelength defining choppers. Optional, default `0.49 m`
            chopper_phase (`sc.Variable`, optional): Phase offset between chopper pulse and ToF zero. Optional, default `-8.`.
            wavelength_cut (`sc.Variable`, optional): Minimum cutoff for wavelength. Optional, default `2.4 Å`.

        Attributes:
            tau (`sc.Variable`): Half of the inverse of the chopper speed.
        """
        if not isinstance(data, sc._scipp.core.DataArray):
            self.data_file = data
            data = scn.load_nexus(self.data_file)
        else:
            self.data_file = data_file
        super().__init__(
            data,
            sample_angle_offset=sample_angle_offset,
            gravity=gravity,
            beam_size=beam_size,
            sample_size=sample_size,
            detector_spatial_resolution=detector_spatial_resolution,
        )
        self.tau = 1 / (2 * chopper_speed)
        self.chopper_detector_distance = chopper_detector_distance
        self.chopper_chopper_distance = chopper_chopper_distance
        self.chopper_phase = chopper_phase
        self.wavelength_cut = wavelength_cut
        self.data.attrs["source_position"] = sc.geometry.position(
            0.0 * sc.units.m, 0.0 * sc.units.m, -chopper_sample_distance)
        self.tof_correction()
        self.data.coords[
            "sigma_lambda_by_lambda"] = self.chopper_chopper_distance / (
                sc.geometry.z(self.data.coords["position"]) -
                sc.geometry.z(self.data.attrs["source_position"]))
        self.find_wavelength()
        self.find_theta()
        self.illumination()
        self.binned = self.wavelength_theta_bin()
        self.find_qz()
        self._setup_orso(reduction_creator, reduction_creator_affiliation,
                         sample_description, data_owner, experiment_id,
                         experiment_date, notebook_file)

    def _setup_orso(self, reduction_creator, reduction_creator_affiliation,
                    sample_description, data_owner, experiment_id,
                    experiment_date, notebook_file):
        experiment = orso.Experiment(self.data.attrs['instrument_name'].value,
                                     'neutron')
        self.orso.data_source.experiment = experiment
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
                'fractional description of sigma from Gaussian resolution function'
            ))
        if reduction_creator is not None:
            self.orso.creator.name = reduction_creator
        if reduction_creator_affiliation is not None:
            self.orso.creator.affiliation = reduction_creator_affiliation
        if sample_description is not None:
            self.orso.data_source.experiment.sample = orso.Sample(
                sample_description)
        if data_owner is not None:
            self.orso.data_source.owner = data_owner
        if experiment_id is not None:
            self.orso.data_source.experiment_id = experiment_id
        if experiment_date is not None:
            self.orso.data_source.experiment_date = experiment_date
        if notebook_file is not None:
            self.orso.reduction.script = notebook_file
        try:
            self.orso.reduction.input_files = orso.Files(
                [orso.File(self.data_file)], [None])
        except TypeError:
            self.orso.reduction.input_files = orso.Files([None], [None])
            self.orso.reduction.comment = 'Live reduction'

    def tof_correction(self):
        """
        Here we correct for the presence of the chopper with respect to the "true" ToF.
        """
        self.data.coords["position"].unit = sc.units.m
        buf = self.data.bins.constituents["data"]
        tof = sc.to_unit(buf.coords["tof"].astype(sc.dtype.float64), 'us')
        tof.unit = sc.units.us
        del buf.coords["tof"]
        buf.coords["tof"] = tof
        tof_offset = self.tau * self.chopper_phase / (180.0 * sc.units.deg)
        tof_cut = self.wavelength_cut * self.chopper_detector_distance / HDM
        tof_e = (sc.Variable(
            values=np.remainder(
                (tof - tof_cut + self.tau).values, self.tau.values),
            unit=sc.units.us,
            dims=["event"],
        ) + tof_cut + tof_offset)
        buf = self.data.bins.constituents["data"]
        tof = tof_e.astype(sc.dtype.float64)
        del buf.coords["tof"]
        buf.coords["tof"] = tof

    def wavelength_masking(self, wavelength_min=None, wavelength_max=None):
        """
        Overwriting the :py:class:`ReflData` wavelength masking functionality.

        Args:
            wavelength_min (`sc.Variable`, optional): Minimum wavelength to be used. Optional, default to `wavelength_cut` value.
            wavelength_max (`sc.Variable`, optional): Maximum wavelength to be used. Optional, default to `wavelength_min + tau * (HDM / chopper_detector_distance)`.
        """
        if wavelength_min is None:
            wavelength_min = self.wavelength_cut
        if wavelength_max is None:
            wavelength_max = wavelength_min + self.tau * (
                HDM / self.chopper_detector_distance)
            if (wavelength_max > sc.max(
                    self.event.coords["wavelength"])).value:
                wavelength_max = sc.max(self.event.coords["wavelength"])
        wavelength_max = sc.to_unit(wavelength_max,
                                    self.event.coords['wavelength'].unit)
        wavelength_min = sc.to_unit(wavelength_min,
                                    self.event.coords['wavelength'].unit)
        self.data.bins.masks["wavelength"] = (
            self.data.bins.coords["wavelength"] < wavelength_min) | (
                self.data.bins.coords["wavelength"] > wavelength_max)


class AmorReference(AmorData):
    """
    Additional functionality of the reference datasets
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
                 data_file=None):
        """
        Args:
            data (`scipp._scipp.core.DataArray`): The data to be reduced.
            sample_angle_offset (`scipp.Variable`, optional): Correction for omega or possibly misalignment of sample. Optional, default `0 degrees of arc`.
            gravity (`bool`, optional): Should gravity be accounted for. Optional, default `True`.
            beam_size (`sc.Variable`, optional): Size of the beam perpendicular to the scattering surface. Optional, default `0.001 m`.
            sample_size (`sc.Variable`, optional): Size of the sample in direction of the beam. Optional, default `0.01 m`.
            detector_spatial_resolution (`sc.Variable`, optional): Spatial resolution of the detector. Optional, default `2.5 mm`
            chopper_sample_distance (`sc.Variable`, optional): Distance from chopper to sample. Optional, default `15.*sc.units.m,`
            chopper_speed (`sc.Variable`, optional): Rotational velocity of the chopper. Optional, default `6.6666... e-6 µs^{-1}`.
            chopper_detector_distance (`sc.Variable`, optional): Distance from chopper to detector. Optional, default `19 m`.
            chopper_chopper_distance (`sc.Variable`, optional): The distance between the wavelength defining choppers. Optional, default `0.49 m`
            chopper_phase (`sc.Variable`, optional): Phase offset between chopper pulse and ToF zero. Optional, default `-8.`.
            wavelength_cut (`sc.Variable`, optional): Minimum cutoff for wavelength. Optional, default `2.4 Å`.
            m_value (`sc.Variable`, optional): m-value of supermirror for reference. Optional, default `5`.

        Attributes:
            tau (`sc.Variable`): Half of the inverse of the chopper speed.
        """
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
        )
        self.m_value = m_value
        supermirror_min_q = ((0.022) * self.event.coords['qz'].unit)
        supermirror_max_q = self.m_value * supermirror_min_q
        self.event.coords['normalisation'] = sc.ones(
            dims=['event'], shape=self.event.data.shape)
        self.event.masks[
            'normalisation'] = self.event.coords['qz'] >= supermirror_max_q
        self.event.coords['normalisation'].values[
            self.event.coords['qz'].values < supermirror_max_q.
            value] = self.event.coords['normalisation'].values[
                self.event.coords['qz'].values < supermirror_max_q.value] / (
                    1.0 - (0.25 / 0.088) * (self.event.coords['qz'].values[
                        self.event.coords['qz'].values <
                        supermirror_max_q.value] - supermirror_min_q.value))
        self.event.coords['normalisation'].values[
            self.event.coords['qz'].values < supermirror_min_q.value] = 1
        self.data.bins.constituents[
            'data'].data = self.event.data / self.event.coords[
                'normalisation'].astype(sc.dtype.float32)


class Normalisation:
    """
    Perform normalisation between a sample and a reference measurement.
    """
    def __init__(self, sample, reference):
        """
        Args:
            sample (`ess.reflectometry.ReflData` or `ess.reflectometry.AmorData`): The sample to be normalised.
            reference (`ess.reflectometry.AmorReference`): The reference measurement to normalise to.
        """
        self.sample = sample
        self.reference = reference
        if self.reference.data_file is None:
            self.sample.orso.reduction.comment += 'live reduction'
        else:
            self.sample.orso.reduction.input_files.reference_files = [
                orso.File(self.reference.data_file)
            ]

    def q_bin(self, bins=None, unit=(1 / sc.units.angstrom).unit):
        """
        Perform q-binned normalisation.

        Args:
            bins (`array_like`, optional): q-bin edges. Defaults to the minimum q-range available with 100 bins.
            unit (`scipp._scipp.core.Unit`, optional): Unit for q. Defaults to 1/Å.

        Returns:
            (`scipp._scipp.core.DataArray`): Normalised data array binned into qz with resolution.
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
                bins = np.linspace(min_q.value, max_q.value, 200)
            binned_sample = self.sample.q_bin(bins, unit)
            binned_reference = self.reference.q_bin(bins, unit)
            del binned_reference.coords['sigma_qz_by_qz']
        else:
            raise sc.NotFoundError("qz coordinate cannot be found.")
        return binned_sample.bins.sum() / binned_reference.bins.sum()

    def wavelength_theta_bin(self, bins):
        """
        Perform wavelength/theta-binned normalisation.

        Args:
            bins (`tuple` of `array_like`): wavelength and theta edges.

        Returns:
            (`scipp._scipp.core.DataArray`): Normalised data array binned into wavelength and theta.
        """
        return self.sample.wavelength_theta_bin(bins).bins.sum(
        ) / self.reference.wavelength_theta_bin(bins).bins.sum()

    def write(self, filename, q_bin_kwargs=None):
        """
        Write the reflectometry intensity data to a file.

        Args:
            filename (`str`): The file path for the file to be saved to.
            q_bin_kwargs (`dict`, optional): A dictionary of keyword arguments to be passed to the :py:func:`q_bin` class method. Optional, default is that default :py:func:`q_bin` keywords arguments are used.
        """
        if q_bin_kwargs is None:
            binned = self.q_bin()
        else:
            binned = self.q_bin(**q_bin_kwargs)
        q_z_edges = binned.coords["qz"].values
        q_z_vector = q_z_edges[:-1] + np.diff(q_z_edges)
        dq_z_vector = binned.coords["sigma_qz_by_qz"].values
        intensity = binned.data.values
        dintensity = np.sqrt(binned.data.variances)
        try:
            header = str(self.sample.orso)
        except AttributeError:
            header = ''
        np.savetxt(filename,
                   np.array([q_z_vector, intensity, dintensity,
                             dq_z_vector]).T,
                   fmt='%.16e',
                   header=header)
