"""
This is the super-class for raw reflectometry data for reduction.
From this super-class, instrument specfic sub-classes may be created (see for example the `AmorData` class)>
"""

# author: Andrew R. McCluskey (arm61)

import numpy as np
import scipp as sc
import scippneutron as scn
from ess.reflectometry import corrections, resolution


class ReflData:
    """
    The general reflectometry data class.
    This will be used by the instrument specific sub-class for data storage, and in essence define the data present reduced data.
    """
    def __init__(
        self,
        data,
        sample_angle_offset=0 * sc.units.deg,
        gravity=True,
        beam_size=0.001 * sc.units.m,
        sample_size=0.01 * sc.units.m,
        detector_spatial_resolution=0.0025 * sc.units.m,
    ):
        """
        Args:
            data (`scipp._scipp.core.DataArray`): The data to be reduced.
            sample_angle_offset (`scipp.Variable`, optional): Correction for omega or possibly misalignment of sample. Optional, default `0 degrees of arc`.
            gravity (`bool`, optional): Should gravity be accounted for. Optional, default `True`.
            beam_size (`sc.Variable`, optional): Size of the beam perpendicular to the scattering surface. Optional, default `0.001 m`.
            sample_size (`sc.Variable`, optional): Size of the sample in direction of the beam. Optional, default `0.01 m`.
            detector_spatial_resolution (`sc.Variable`, optional): Spatial resolution of the detector. Optional, default `2.5 mm`
        """
        self.data = data
        self.data.bins.constituents["data"].variances = np.ones_like(
            self.data.bins.constituents["data"].values)
        self.sample_angle_offset = sample_angle_offset
        self.gravity = gravity
        self.beam_size = beam_size
        self.sample_size = sample_size
        self.detector_spatial_resolution = detector_spatial_resolution

    @property
    def event(self):
        """
        Return the event data buffer.

        Returns
            (`scipp._scipp.core.DataArrayView`): Event data information.
        """
        return self.data.bins.constituents["data"]

    def q_bin(self, bins=None, unit=(1 / sc.units.angstrom).unit):
        """
        Return data that has been binned in the q-bins passed.

        Args:
            bins (`array_like`): q-bin edges.
            unit (`scipp._scipp.core.Unit`): Unit for q. Defaults to 1/Å.

        Returns:
            (`scipp._scipp.core.DataArray`): Data array binned into qz with resolution.
        """
        if "qz" in self.event.coords and "tof" in self.event.coords:
            if bins is None:
                q_z_vector = self.event.coords["qz"].values
                bins = np.linspace(q_z_vector.min(), q_z_vector.max(), 200)
            edges = sc.array(dims=["qz"], unit=unit, values=bins)
            binned = sc.bin(self.data,
                            erase=["detector_id", "tof"],
                            edges=[edges])
            if "sigma_qz_by_qz" in self.event.coords:
                qzr = []
                for i in binned.data.values:
                    try:
                        qzr.append(i.coords["sigma_qz_by_qz"].values.max())
                    except ValueError:
                        qzr.append(0)
                qzr = np.array(qzr)
                binned.coords["sigma_qz_by_qz"] = sc.Variable(values=qzr,
                                                              dims=["qz"])
        else:
            raise sc.NotFoundError("qz, or tof coordinate cannot be found.")
        return binned

    def wavelength_theta_bin(self,
                             bins=None,
                             units=(sc.units.angstrom, sc.units.deg)):
        """
        Return data that has been binned in the wavelength and theta bins passed.

        Args:
            bins (`tuple` of `array_like`): wavelength and theta edges.
            unit (`tuple` of `scipp._scipp.core.Unit`): Units for wavelength and theta. Defaults to [Å, deg].

        Returns:
            (`scipp._scipp.core.DataArray`): Data array binned into wavelength and theta.
        """
        if bins is None:
            wavelength = self.event.coords["wavelength"].values
            theta = self.event.coords["theta"].values
            bins = [
                np.linspace(wavelength.min(), wavelength.max(), 100),
                np.linspace(theta.min(), theta.max(), 100),
            ]
        wavelength_bins = sc.array(dims=["wavelength"],
                                   unit=units[0],
                                   values=bins[0])
        theta_bins = sc.array(dims=["theta"], unit=units[1], values=bins[1])
        return sc.bin(
            self.data,
            erase=["detector_id", "tof"],
            edges=[theta_bins, wavelength_bins],
        )

    def q_theta_bin(self,
                    bins=None,
                    units=((1 / sc.units.angstrom).unit, sc.units.deg)):
        """
        Return data that has been binned in the wavelength and theta bins passed.

        Args:
            bins (`tuple` of `array_like`): wavelength and theta edges.
            unit (`tuple` of `scipp._scipp.core.Unit`): Units for wavelength and theta. Defaults to [Å, deg].

        Returns:
            (`scipp._scipp.core.DataArray`): Data array binned into wavelength and theta.
        """
        if bins is None:
            q_z_vector = self.event.coords["qz"].values
            theta = self.event.coords["theta"].values
            bins = [
                np.linspace(q_z_vector.min(), q_z_vector.max(), 100),
                np.linspace(theta.min(), theta.max(), 100),
            ]
        q_bins = sc.array(dims=["qz"], unit=units[0], values=bins[0])
        theta_bins = sc.array(dims=["theta"], unit=units[1], values=bins[1])
        return sc.bin(self.data,
                      erase=["detector_id", "tof"],
                      edges=[theta_bins, q_bins])

    def wavelength_q_bin(
            self,
            bins=None,
            units=(sc.units.angstrom, (1 / sc.units.angstrom).unit),
    ):
        """
        Return data that has been binned in the wavelength and theta bins passed.

        Args:
            bins (`tuple` of `array_like`): wavelength and theta edges.
            unit (`tuple` of `scipp._scipp.core.Unit`): Units for wavelength and theta. Defaults to [Å, deg].

        Returns:
            (`scipp._scipp.core.DataArray`): Data array binned into wavelength and theta.
        """
        if bins is None:
            wavelength = self.event.coords["wavelength"].values
            q_z_vector = self.event.coords["qz"].values
            bins = [
                np.linspace(wavelength.min(), wavelength.max(), 100),
                np.linspace(q_z_vector.min(), q_z_vector.max(), 100),
            ]
        wavelength_bins = sc.array(dims=["wavelength"],
                                   unit=units[0],
                                   values=bins[0])
        q_bins = sc.array(dims=["qz"], unit=units[1], values=bins[1])
        return sc.bin(
            self.data,
            erase=["detector_id", "tof"],
            edges=[q_bins, wavelength_bins],
        )

    def find_wavelength(self):
        """
        From the time-of-flight data, find the wavelength for each neutron event.
        """
        self.data.bins.constituents["data"].coords["wavelength"] = (
            scn.convert(
                self.data, "tof", "wavelength",
                scatter=True).bins.constituents["data"].coords["wavelength"])

    def find_theta(self):
        """
        Calculate the theta angle (and resolution) with or without a gravity correction.
        """
        if self.gravity:
            nu_angle = corrections.angle_with_gravity(
                self.data,
                self.data.coords["position"],
                self.data.attrs["sample_position"],
            )
            theta = self.sample_angle_offset + nu_angle
            self.data.bins.constituents["data"].coords["theta"] = theta
            # Check if the beam size on the sample is overilluminating the
            # sample. Using the beam on sample size from the value of theta
            # where the neutrons scatter from a point
            half_beam_on_sample = (corrections.illumination_of_sample(
                self.beam_size, self.sample_size, theta) / 2.0)
            # Find the range of possible positions that the neutron could
            # strike, this range of theta values is taken to be the full
            # width half maximum for the theta distribution
            self.data.bins.constituents["data"].attrs[
                "offset_postive"] = resolution.z_offset(
                    self.data.attrs["sample_position"], half_beam_on_sample)
            self.data.bins.constituents["data"].attrs[
                "offset_negative"] = resolution.z_offset(
                    self.data.attrs["sample_position"], -half_beam_on_sample)
            angle_max = corrections.angle_with_gravity(
                self.data,
                self.data.coords["position"],
                self.data.bins.attrs["offset_postive"],
            )
            angle_min = corrections.angle_with_gravity(
                self.data,
                self.data.coords["position"],
                self.data.bins.attrs["offset_negative"],
            )
            del self.data.bins.constituents["data"].attrs["offset_postive"]
            del self.data.bins.constituents["data"].attrs["offset_negative"]
            sigma_theta_position = (angle_max - angle_min) / 2.354820045
            self.data.bins.constituents["data"].attrs[
                "sigma_theta_position"] = sigma_theta_position
            # Then find the full width half maximum of the theta distribution
            # due to the detector's spatial resolution, which we will call
            # sigma_gamma
            sigma_gamma = resolution.detector_resolution(
                self.detector_spatial_resolution,
                sc.geometry.z(self.data.coords["position"]),
                sc.geometry.z(self.data.attrs["sample_position"]),
            )
            self.data.attrs["sigma_gamma"] = sigma_gamma
            sigma_theta = sc.sqrt(
                (self.data.attrs["sigma_gamma"] /
                 self.data.bins.coords["theta"]) *
                (self.data.attrs["sigma_gamma"] /
                 self.data.bins.coords["theta"]) +
                (self.data.bins.attrs["sigma_theta_position"] /
                 self.data.bins.coords["theta"]) *
                (self.data.bins.attrs["sigma_theta_position"] /
                 self.data.bins.coords["theta"]))
            self.data.bins.constituents["data"].coords[
                "sigma_theta_by_theta"] = sigma_theta.bins.constituents["data"]
        else:
            raise NotImplementedError

    def find_qz(self):
        """
        Calculate the scattering vector (and resolution).
        """
        self.data.bins.constituents["data"].coords["qz"] = (
            4.0 * np.pi *
            sc.sin(self.data.bins.constituents["data"].coords["theta"]) /
            self.data.bins.constituents["data"].coords["wavelength"])
        self.data.coords["s_qz_bins"] = sc.zeros(
            dims=self.data.coords["detector_id"].dims,
            shape=self.data.coords["detector_id"].shape,
        )
        for i in list(self.data.coords.keys()):
            if "sigma" in i:
                self.data.coords["s_qz_bins"] += (self.data.coords[i] *
                                                  self.data.coords[i])
        self.data.bins.constituents["data"].coords["s_qz_events"] = sc.zeros(
            dims=self.data.bins.constituents["data"].coords["qz"].dims,
            shape=self.data.bins.constituents["data"].coords["qz"].shape,
        )
        for i in list(self.data.bins.constituents["data"].coords.keys()):
            if "sigma" in i:
                self.data.bins.constituents["data"].coords["s_qz_events"] += (
                    self.data.bins.coords[i] *
                    self.data.bins.coords[i]).bins.constituents["data"]
        self.data.bins.constituents["data"].coords["sigma_qz_by_qz"] = sc.sqrt(
            (self.data.coords["s_qz_bins"] +
             self.data.bins.coords["s_qz_events"]).bins.constituents["data"])

    def illumination(self):
        """
        Perform illumination correction.
        """
        self.event.coords["illumination"] = sc.Variable(
            values=corrections.illumination_correction(
                self.beam_size,
                self.sample_size,
                self.event.coords["theta"],
            ),
            dims=["event"],
        )
        self.data /= self.data.bins.coords["illumination"]

    def detector_masking(
        self,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        z_min=None,
        z_max=None,
    ):
        """
        Masking on detector pixels.

        Args:
            x_min (`sc.Variable`, optional): Minimum x-dimension to be used. Optional, default no minimum mask.
            x_max (`sc.Variable`, optional): Maximum x-dimension to be used. Optional, default no maximum mask.
            y_min (`sc.Variable`, optional): Minimum y-dimension to be used. Optional, default no minimum mask.
            y_max (`sc.Variable`, optional): Maximum y-dimension to be used. Optional, default no maximum mask.
            z_min (`sc.Variable`, optional): Minimum z-dimension to be used. Optional, default no minimum mask.
            z_max (`sc.Variable`, optional): Maximum z-dimension to be used. Optional, default no maximum mask.
        """
        x_position = sc.geometry.x(self.data.coords["position"])
        y_position = sc.geometry.y(self.data.coords["position"])
        z_position = sc.geometry.z(self.data.coords["position"])
        if x_min is None:
            x_min = sc.min(x_position)
        if x_max is None:
            x_max = sc.max(x_position)
        if y_min is None:
            y_min = sc.min(y_position)
        if y_max is None:
            y_max = sc.max(y_position)
        if z_min is None:
            z_min = sc.min(z_position)
        if z_max is None:
            z_max = sc.max(z_position)
        self.data.masks["x_mask"] = (x_position < x_min) | (x_position > x_max)
        self.data.masks["y_mask"] = (y_position < y_min) | (y_position > y_max)
        self.data.masks["z_mask"] = (z_position < z_min) | (z_position > z_max)

    def theta_masking(self, theta_min=None, theta_max=None):
        """
        Masking data based on reflected angle.

        Args:
            theta_min (`sc.Variable`, optional): Minimum theta to be used. Optional, default no minimum mask.
            theta_max (`sc.Variable`, optional): Maximum theta to be used. Optional, default no maximum mask.
        """
        if theta_min is None:
            theta_min = 0 * sc.units.deg
        if theta_max is None:
            theta_max = 180 * sc.units.deg
        try:
            self.data.masks["theta"] = (self.data.coords["theta"] <
                                        theta_min) | (self.data.coords["theta"]
                                                      > theta_max)
        except sc.NotFoundError:
            self.data.bins.masks["theta"] = (
                self.data.bins.coords["theta"] <
                theta_min) | (self.data.bins.coords["theta"] > theta_max)

    def wavelength_masking(self, wavelength_min=None, wavelength_max=None):
        """
        Masking data based on wavelength.

        Args:
            wavelength_min (`sc.Variable`, optional): Minimum wavelength to be used. Optional, default no minimum mask.
            wavelength_max (`sc.Variable`, optional): Maximum wavelength to be used. Optional, default no maximum mask.
        """
        if wavelength_min is None:
            wavelength_min = sc.min(self.event.coords["wavelength"])
        if wavelength_max is None:
            wavelength_max = sc.max(self.event.coords["wavelength"])
        self.data.bins.masks["wavelength"] = (
            self.data.bins.coords["wavelength"] < wavelength_min) | (
                self.data.bins.coords["wavelength"] > wavelength_max)

    def write(self, filename, q_bin_kwargs=None):
        """
        Write the reflectometry intensity data to a file.

        Args:
            filename (`str`): The file path for the file to be saved to.
            q_bin_kwargs (`dict`, optional): A dictionary of keyword arguments to be passed to the :py:func:`q_bin` class method. Optional, default is that default :py:func:`q_bin` keywords arguments are used.
        """
        if q_bin_kwargs is None:
            binned = self.q_bin().bins.sum()
        else:
            binned = self.q_bin(**q_bin_kwargs).bins.sum()
        q_z_edges = binned.coords["qz"].values
        q_z_vector = q_z_edges[:-1] + np.diff(q_z_edges)
        dq_z_vector = binned.coords["sigma_qz_by_qz"].values
        intensity = binned.data.values
        dintensity = np.sqrt(binned.data.variances)
        np.savetxt(
            filename,
            np.array([q_z_vector, intensity, dintensity, dq_z_vector]).T,
        )
