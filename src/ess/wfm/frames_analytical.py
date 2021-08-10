# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
from typing import Union
from .tools import frame_opening_and_closing_times


def frames_analytical(data: Union[sc.DataArray, sc.Dataset],
                      wfm_chopper_names: list = ["WFMC1", "WFMC2"]) -> sc.Dataset:
    """
    Compute analytical frame boundaries and shifts based on chopper
    parameters and detector pixel positions.
    A set of frame boundaries is returned for each pixel.
    The frame shifts are the same for all pixels.
    See Schmakat et al. (2020);
    https://www.sciencedirect.com/science/article/pii/S0168900220308640
    for a description of the procedure.

    TODO: This currently ignores scattering paths, only the distance from
    source to pixel.
    For imaging, this is what we want, but for scattering techniques, we should
    use l1 + l2 in the future.
    """

    # Compute distances for each pixel
    pos_norm = sc.norm(data.meta["position"])

    # Get the number of WFM frames
    nframes = data.meta["choppers"].value["opening_angles_open"].sizes["frame"]

    # Now find frame boundaries
    frames = sc.Dataset()
    frames["time_min"] = sc.zeros(dims=["frame"] + data.meta["position"].dims,
                                  shape=[nframes] + data.meta["position"].shape,
                                  unit=sc.units.us)
    frames["time_max"] = sc.zeros_like(frames["time_min"])
    frames["delta_time_min"] = sc.zeros_like(frames["time_min"])
    frames["delta_time_max"] = sc.zeros_like(frames["time_min"])
    frames["wavelength_min"] = sc.zeros_like(frames["time_min"])
    frames["wavelength_min"].unit = sc.units.angstrom
    frames["wavelength_max"] = sc.zeros_like(frames["wavelength_min"])
    frames["delta_wavelength_min"] = sc.zeros_like(frames["wavelength_min"])
    frames["delta_wavelength_max"] = sc.zeros_like(frames["wavelength_min"])

    frames["time_correction"] = sc.zeros(dims=["frame"],
                                         shape=[nframes],
                                         unit=sc.units.us)

    # Identify the position of the WFM choppers in the array of choppers
    chopper_names = list(data.meta["choppers"].value["names"].values)
    wfm_chopper_indices = [chopper_names.index(name) for name in wfm_chopper_names]

    # Order the WFM chopper indices from the chopper closest to the source to the
    # furthest away
    wfm_chopper_distances = [
        sc.norm(data.meta["choppers"].value["position"]["chopper", ind].data).value
        for ind in wfm_chopper_indices
    ]
    wfm_chopper_indices = np.array(wfm_chopper_indices)[np.argsort(
        wfm_chopper_distances)]

    near_wfm_chopper_position = data.meta["choppers"].value["position"][
        "chopper", wfm_chopper_indices[0]].data
    far_wfm_chopper_position = data.meta["choppers"].value["position"][
        "chopper", wfm_chopper_indices[1]].data

    # Distance between WFM choppers
    dz_wfm = sc.norm(far_wfm_chopper_position - near_wfm_chopper_position)
    # Mid-point between WFM choppers
    z_wfm = 0.5 * sc.norm(near_wfm_chopper_position + far_wfm_chopper_position)
    # Ratio of WFM chopper distances
    z_ratio_wfm = (sc.norm(far_wfm_chopper_position) /
                   sc.norm(near_wfm_chopper_position))

    # Neutron mass to Planck constant ratio
    # TODO: would be nice to use physical constants in scipp or scippneutron
    alpha = 2.5278e+2 * (sc.Unit('us') / sc.Unit('angstrom') / sc.Unit('m'))

    # Now compute frames opening and closing edges at the detector positions
    for i in range(nframes):

        # Get frame parameters
        tstart, tend = frame_opening_and_closing_times(
            data.meta["choppers"].value["frame", i])

        # Frame time corrections: these are the mid-time point between the WFM choppers,
        # which is the same as the opening edge of the second WFM chopper in the case
        # of optically blind choppers.
        frames["time_correction"]["frame", i] = tstart["chopper",
                                                       wfm_chopper_indices[1]]

        # Find delta_t for the min and max wavelengths:
        # dt_lambda_max is equal to the time width of the WFM choppers windows
        dt_lambda_max = tend['chopper',
                             wfm_chopper_indices[0]] - tstart['chopper',
                                                              wfm_chopper_indices[0]]

        # t_lambda_max is found from the relation between t and delta_t: equation (2) in
        # Schmakat et al. (2020).
        t_lambda_max = (dt_lambda_max / dz_wfm) * (pos_norm - z_wfm)

        # t_lambda_min is found from the relation between lambda_N and lambda_N+1,
        # equation (3) in Schmakat et al. (2020).
        t_lambda_min = t_lambda_max * z_ratio_wfm - data.meta["pulse_length"] * (
            (pos_norm - z_wfm) / sc.norm(near_wfm_chopper_position))

        # dt_lambda_min is found from the relation between t and delta_t: equation (2)
        # in Schmakat et al. (2020).
        dt_lambda_min = dz_wfm * t_lambda_min / (pos_norm - z_wfm)

        # Compute wavelength information
        lambda_min = t_lambda_min / (alpha * (pos_norm - z_wfm))
        lambda_max = t_lambda_max / (alpha * (pos_norm - z_wfm))
        dlambda_min = dz_wfm * lambda_min / (pos_norm - z_wfm)
        dlambda_max = dz_wfm * lambda_max / (pos_norm - z_wfm)

        # Frame edges and resolutions for each pixel.
        # The frames do not stop at t_lambda_min and t_lambda_max, they also include the
        # fuzzy areas (delta_t) at the edges.
        frames["time_min"][
            "frame",
            i] = t_lambda_min - 0.5 * dt_lambda_min + frames["time_correction"]["frame",
                                                                                i]
        frames["delta_time_min"]["frame", i] = dt_lambda_min

        frames["time_max"][
            "frame",
            i] = t_lambda_max + 0.5 * dt_lambda_max + frames["time_correction"]["frame",
                                                                                i]
        frames["delta_time_max"]["frame", i] = dt_lambda_max
        frames["wavelength_min"]["frame", i] = lambda_min
        frames["wavelength_max"]["frame", i] = lambda_max
        frames["delta_wavelength_min"]["frame", i] = dlambda_min
        frames["delta_wavelength_max"]["frame", i] = dlambda_max

    frames["wfm_chopper_mid_point"] = sc.mean(
        sc.concatenate(data.meta["choppers"].value["position"]["chopper", 0],
                       data.meta["choppers"].value["position"]["chopper", 1], 'none'))

    return frames
