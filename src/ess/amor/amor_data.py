"""
This is the class for data reduction from the Amor instrument, which is a subclass of the broader `ReflData` class.
Features of this class included correcting for the time-of-flight measurement at Amor.
"""
import scipp as sc
import numpy as np
from ess.reflectometry import HDM
from ess.reflectometry.data import ReflData


class AmorData(ReflData):
    """
    Reduction of AMOR data.
    """
    def __init__(
        self,
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
        super().__init__(
            data,
            sample_angle_offset,
            gravity,
            beam_size,
            sample_size,
            detector_spatial_resolution,
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
        range = [
            sc.min(self.event.coords['wavelength']).value,
            wavelength_min.value, wavelength_max.value,
            sc.max(self.event.coords['wavelength']).value
        ]
        wavelength = sc.array(dims=['wavelength'],
                              unit=self.event.coords['wavelength'].unit,
                              values=range)
        self.data = sc.bin(self.data, edges=[wavelength])
        self.data.masks['wavelength'] = sc.array(dims=['wavelength'],
                                                 values=[True, False, True])


class AmorReference(AmorData):
    """
    Additional functionality of the reference datasets
    """
    def __init__(
        self,
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
            m_value (`sc.Variable`, optional): m-value of supermirror for reference. Optional, default `5`.

        Attributes:
            tau (`sc.Variable`): Half of the inverse of the chopper speed.
        """
        super().__init__(
            data,
            sample_angle_offset,
            gravity,
            beam_size,
            sample_size,
            detector_spatial_resolution,
            chopper_sample_distance,
            chopper_speed,
            chopper_detector_distance,
            chopper_chopper_distance,
            chopper_phase,
            wavelength_cut,
        )
        self.m_value = m_value
        supermirror_min_q = ((0.022) * self.event.coords['qz'].unit)
        supermirror_max_q = self.m_value * supermirror_min_q
        self.event.coords['normalisation'] = sc.ones(
            dims=['event'], shape=self.event.data.shape
        ) * self.event.data.shape[0] / self.event.data.shape[0]
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

    def q_bin(self, bins=None, unit=(1 / sc.units.angstrom).unit):
        """
        Perform q-binned normalisation.

        Args:
            bins (`array_like`, optional): q-bin edges. Defaults to the minimum q-range available with 100 bins.
            unit (`scipp._scipp.core.Unit`, optional): Unit for q. Defaults to 1/Å.

        Returns:
            (`scipp._scipp.core.DataArray`): Normalised data array binned into qz with resolution.
        """
        if bins is None:
            min_q = np.max([
                sc.min(self.sample.event.coords['qz']).value,
                sc.min(self.reference.event.coords['qz']).value
            ]) * self.sample.event.coords['qz'].unit
            max_q = np.min([
                sc.max(self.sample.event.coords['qz']).value,
                sc.max(self.reference.event.coords['qz']).value
            ]) * self.sample.event.coords['qz'].unit
            bins = np.linspace(min_q.value, max_q.value, 100)
        binned_sample = self.sample.q_bin(bins, unit).copy()
        binned_reference = self.reference.q_bin(bins, unit).copy()
        del binned_reference.coords['sigma_qz_by_qz']
        return binned_sample.bins.sum() / binned_reference.bins.sum()

    def wavelength_theta_bin(self, bins):
        """
        Perform wavelength/theta-binned normalisation.


        """
        return self.sample.wavelength_theta_bin(bins).bins.sum(
        ) / self.reference.wavelength_theta_bin(bins).bins.sum()
