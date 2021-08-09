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
    frames["left_edges"] = sc.zeros(dims=["frame"] + data.meta["position"].dims,
                                    shape=[nframes] + data.meta["position"].shape,
                                    unit=sc.units.us)
    frames["right_edges"] = sc.zeros_like(frames["left_edges"])
    frames["left_dt"] = sc.zeros_like(frames["left_edges"])
    frames["right_dt"] = sc.zeros_like(frames["left_edges"])
    frames["shifts"] = sc.zeros(dims=["frame"], shape=[nframes], unit=sc.units.us)

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

    # Now compute frames opening and closing edges at the detector positions
    for i in range(nframes):

        # Get frame parameters
        tstart, tend = frame_opening_and_closing_times(
            data.meta["choppers"].value["frame", i])

        # Frame shifts: these are the mid-time point between the WFM choppers,
        # which is the same as the opening edge of the second WFM chopper in the case
        # of optically blind choppers.
        frames["shifts"]["frame", i] = tstart["chopper", wfm_chopper_indices[1]]

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

        # Frame edges and resolutions for each pixel.
        # The frames do not stop at t_lambda_min and t_lambda_max, they also include the
        # fuzzy areas (delta_t) at the edges.
        frames["left_edges"][
            "frame",
            i] = t_lambda_min - 0.5 * dt_lambda_min + frames["shifts"]["frame", i]
        frames["left_dt"]["frame", i] = dt_lambda_min

        frames["right_edges"][
            "frame",
            i] = t_lambda_max + 0.5 * dt_lambda_max + frames["shifts"]["frame", i]
        frames["right_dt"]["frame", i] = dt_lambda_max

    frames["wfm_chopper_mid_point"] = sc.mean(
        sc.concatenate(
            data.meta["choppers"].value["position"]["chopper", wfm_chopper_indices[0]],
            data.meta["choppers"].value["position"]["chopper",
                                                    wfm_chopper_indices[1]], 'none'))

    return frames
