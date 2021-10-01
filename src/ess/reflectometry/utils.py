# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippneutron as scn


def to_wavelength(data, wavelength_cut=None):
    """
    From the time-of-flight data, find the wavelength for each neutron event.
    """
    data = scn.convert(data, origin='tof', target='wavelength', scatter=True)
    # Select desired wavelength range
    if wavelength_cut is not None:
        data = sc.bin(data,
                      edges=[
                          sc.concatenate(wavelength_cut,
                                         data.coords['wavelength'].max(), 'wavelength')
                      ])


def find_theta(self):
    """
    Calculate the theta angle (and resolution) with or without a gravity correction.
    """
    if gravity:
        nu_angle = corrections.angle_with_gravity(
            data,
            data.meta["position"],
            data.meta["sample_position"],
        )
        theta = -sample_angle_offset + nu_angle
        data.bins.constituents["data"].coords["theta"] = theta
        # Check if the beam size on the sample is overilluminating the
        # sample. Using the beam on sample size from the value of theta
        # where the neutrons scatter from a point
        half_beam_on_sample = (
            corrections.illumination_of_sample(beam_size, sample_size, theta) / 2.0)
        # Find the range of possible positions that the neutron could
        # strike, this range of theta values is taken to be the full
        # width half maximum for the theta distribution
        offset_positive = resolution.z_offset(data.meta["sample_position"],
                                              half_beam_on_sample)
        offset_negative = resolution.z_offset(data.meta["sample_position"],
                                              half_beam_on_sample)
        data.bins.constituents['data'].attrs['offset_positive'] = offset_positive
        data.bins.constituents['data'].attrs['offset_negative'] = offset_negative
        angle_max = corrections.angle_with_gravity(data, data.coords["position"],
                                                   data.bins.attrs['offset_positive'])
        angle_min = corrections.angle_with_gravity(
            data,
            data.coords["position"],
            data.bins.attrs['offset_negative'],
        )
        del data.bins.constituents['data'].attrs['offset_positive']
        del data.bins.constituents['data'].attrs['offset_negative']
        fwhm_to_std = 2 * np.sqrt(2 * np.log(2))
        sigma_theta_position = (angle_max - angle_min) / fwhm_to_std
        data.bins.constituents["data"].attrs[
            "sigma_theta_position"] = sigma_theta_position
        # Then find the full width half maximum of the theta distribution
        # due to the detector's spatial resolution, which we will call
        # sigma_gamma
        sigma_gamma = resolution.detector_resolution(
            detector_spatial_resolution, data.coords["position"].fields.z,
            data.meta["sample_position"].fields.z)
        data.attrs["sigma_gamma"] = sigma_gamma
        sigma_theta = sc.sqrt(
            (data.attrs["sigma_gamma"] / data.bins.coords["theta"]) *
            (data.attrs["sigma_gamma"] / data.bins.coords["theta"]) +
            (data.bins.attrs["sigma_theta_position"] / data.bins.coords["theta"]) *
            (data.bins.attrs["sigma_theta_position"] / data.bins.coords["theta"]))
        data.bins.constituents["data"].coords[
            "sigma_theta_by_theta"] = sigma_theta.bins.constituents["data"]
    else:
        raise NotImplementedError


def find_qz(self):
    """
    Calculate the scattering vector (and resolution).
    """
    data.bins.constituents["data"].coords["qz"] = (
        4.0 * np.pi * sc.sin(data.bins.constituents["data"].coords["theta"]) /
        data.bins.constituents["data"].coords["wavelength"])
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