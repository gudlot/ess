# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
import scippneutron as scn
from . import corrections
from . import resolution
from .constants import HDM


def to_wavelength(data, wavelength_bins=None):
    """
    From the time-of-flight data, find the wavelength for each neutron event.
    """
    data = scn.convert(data, origin='tof', target='wavelength', scatter=True)
    # Select desired wavelength range
    if wavelength_bins is not None:
        data = sc.bin(data, edges=[wavelength_bins])
    return data


def compute_theta(data):
    """
    Calculate the theta angle (and resolution) with or without a gravity correction.
    """
    if data.meta["gravity"].value:
        # Temporary broadcast of position coord to all events
        # data.bins.constituents["data"].coords['position'] = sc.empty(
        #     sizes=data.bins.constituents["data"].sizes,
        #     dtype=sc.dtype.vector_3_float64,
        #     unit='m')
        # data.bins.coords['position'][...] = data.meta['position']

        nu_angle = corrections.angle_with_gravity(
            data.meta["velocity"],
            data.meta["position"],
            data.meta["sample_position"],
        )
        # return {"y_dash": nu_angle}
        theta = -data.meta["sample_angle_offset"] + nu_angle
        # data.bins.constituents["data"].coords["theta"] = theta
        # data.bins.coords["theta"] = theta

        # Check if the beam size on the sample is overilluminating the
        # sample. Using the beam on sample size from the value of theta
        # where the neutrons scatter from a point
        half_beam_on_sample = (corrections.illumination_of_sample(
            data.meta["beam_size"], data.meta["sample_size"], theta) / 2.0)
        # Find the range of possible positions that the neutron could
        # strike, this range of theta values is taken to be the full
        # width half maximum for the theta distribution
        offset_positive = resolution.z_offset(data.meta["sample_position"],
                                              half_beam_on_sample)
        offset_negative = resolution.z_offset(data.meta["sample_position"],
                                              -half_beam_on_sample)
        # data.bins.constituents['data'].attrs['offset_positive'] = offset_positive
        # data.bins.constituents['data'].attrs['offset_negative'] = offset_negative
        # data.bins.attrs['offset_positive'] = offset_positive
        # data.bins.attrs['offset_negative'] = offset_negative
        angle_max = corrections.angle_with_gravity(data.meta["velocity"],
                                                   data.meta["position"],
                                                   offset_positive)
        angle_min = corrections.angle_with_gravity(data.meta["velocity"],
                                                   data.meta["position"],
                                                   offset_negative)
        # del data.bins.constituents['data'].attrs['offset_positive']
        # del data.bins.constituents['data'].attrs['offset_negative']
        fwhm_to_std = 2 * np.sqrt(2 * np.log(2))
        sigma_theta_position = (angle_max - angle_min) / fwhm_to_std
        # data.bins.constituents["data"].attrs[
        #     "sigma_theta_position"] = sigma_theta_position
        # data.bins.attrs["sigma_theta_position"] = sigma_theta_position
        # Then find the full width half maximum of the theta distribution
        # due to the detector's spatial resolution, which we will call
        # sigma_gamma
        sigma_gamma = resolution.detector_resolution(
            data.meta["detector_spatial_resolution"], data.meta["position"].fields.z,
            data.meta["sample_position"].fields.z)
        # data.attrs["sigma_gamma"] = sigma_gamma
        # sigma_theta = sc.sqrt(
        #     (data.attrs["sigma_gamma"] / data.bins.coords["theta"]) *
        #     (data.attrs["sigma_gamma"] / data.bins.coords["theta"]) +
        #     (data.bins.attrs["sigma_theta_position"] / data.bins.coords["theta"]) *
        #     (data.bins.attrs["sigma_theta_position"] / data.bins.coords["theta"]))
        sigma_theta = sc.sqrt((sigma_gamma / theta) * (sigma_gamma / theta) +
                              (sigma_theta_position / theta) *
                              (sigma_theta_position / theta))

        # data.bins.constituents["data"].coords[
        #     "sigma_theta_by_theta"] = sigma_theta.bins.constituents["data"]
        # data.bins.coords["sigma_theta_by_theta"] = sigma_theta.bins.constituents["data"]
        return {
            "theta": theta,
            # "sigma_theta_position": sigma_theta_position,
            # "sigma_gamma": sigma_gamma,
            # "nu_angle": nu_angle,
            "sigma_theta_by_theta": sigma_theta
        }
    else:
        raise NotImplementedError


def compute_qz(theta, wavelength):
    """
    Calculate the scattering vector (and resolution).
    """
    # # data.bins.constituents["data"].coords["qz"] = (
    # #     4.0 * np.pi * sc.sin(data.bins.constituents["data"].coords["theta"]) /
    # #     data.bins.constituents["data"].coords["wavelength"])
    # data.bins.coords["qz"] = (4.0 * np.pi * sc.sin(data.bins.coords["theta"]) /
    #                           data.bins.coords["wavelength"])
    return 4.0 * np.pi * sc.sin(theta) / wavelength
    data.coords["s_qz_bins"] = sc.zeros(
        dims=data.coords["detector_id"].dims,
        shape=data.coords["detector_id"].shape,
    )
    for i in list(data.coords.keys()):
        if "sigma" in i:
            data.coords["s_qz_bins"] += (data.coords[i] * data.coords[i])
    data.bins.constituents["data"].coords["s_qz_events"] = sc.zeros(
        dims=data.bins.constituents["data"].coords["qz"].dims,
        shape=data.bins.constituents["data"].coords["qz"].shape,
    )
    for i in list(data.bins.constituents["data"].coords.keys()):
        if "sigma" in i:
            data.bins.constituents["data"].coords["s_qz_events"] += (
                data.bins.coords[i] * data.bins.coords[i]).bins.constituents["data"]
    data.bins.constituents["data"].coords["sigma_qz_by_qz"] = sc.sqrt(
        (data.coords["s_qz_bins"] +
         data.bins.coords["s_qz_events"]).bins.constituents["data"])


def compute_velocity(wavelength, to_unit='m/s'):
    return sc.to_unit(HDM / wavelength, "m/s")
