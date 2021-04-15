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
        chopper_phase=-8.0 * sc.units.dimensionless,
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
            0.0 * sc.units.m, 0.0 * sc.units.m, -chopper_sample_distance
        )
        self.tof_correction()
        self.data.coords[
            "sigma_lambda_by_lambda"
        ] = self.chopper_chopper_distance / (
            sc.geometry.z(self.data.coords["position"])
            - sc.geometry.z(self.data.attrs["source_position"])
        )
        self.find_tof()
        self.find_theta()
        self.find_qz()
        self.illumination()

    def tof_correction(self):
        """
        Here we correct for the presence of the chopper with respect to the "true" ToF.
        """
        self.data.coords["position"].unit = sc.units.m
        buf = self.data.bins.constituents["data"]
        tof = buf.coords["tof"].astype(sc.dtype.float64) * 1e-3
        tof.unit = sc.units.us
        del buf.coords["tof"]
        buf.coords["tof"] = tof
        tof_offset = self.tau * self.chopper_phase / 180.0
        tof_e = self.data.bins.constituents["data"].coords["tof"]
        tof_cut = self.wavelength_cut * self.chopper_detector_distance / HDM
        tof_e = (
            sc.Variable(
                values=np.remainder(
                    (tof_e - tof_cut + self.tau).values, self.tau.values
                ),
                unit=sc.units.us,
                dims=["event"],
            )
            + tof_cut
            + tof_offset
        )
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
                HDM / self.chopper_detector_distance
            )
        self.data.bins.masks["wavelength"] = (
            self.data.bins.coords["wavelength"] < wavelength_min
        ) | (self.data.bins.coords["wavelength"] > wavelength_max)
