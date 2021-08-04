# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipp as sc
from .tools import get_frame_properties


def frames_analytical(data, wfm_chopper_names=["WFMC1", "WFMC2"]):
    """
    Compute analytical frame boundaries and shifts based on chopper
    parameters and detector pixel positions.
    A set of frame boundaries is returned for each pixel.
    The frame shifts are the same for all pixels.
    See https://www.sciencedirect.com/science/article/pii/S0168900220308640
    for a description of the procedure.

    TODO: This currently ignores scattering paths, only the distance from
    source to pixel.
    For imaging, this is what we want, but for scattering techniques, we should
    use l1 + l2 in the future.
    """

    # Compute distances for each pixel
    pos_norm = sc.norm(data.meta["position"])
    source_pos = sc.norm(data.meta["source_position"])

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

    # Distance between WFM choppers
    dz_wfm = sc.norm(
        data.meta["choppers"].value["position"]["chopper",
                                                wfm_chopper_indices[1]].data -
        data.meta["choppers"].value["position"]["chopper", wfm_chopper_indices[0]].data)
    # Mid-point between WFM choppers
    z_wfm = 0.5 * sc.norm(
        data.meta["choppers"].value["position"]["chopper",
                                                wfm_chopper_indices[0]].data +
        data.meta["choppers"].value["position"]["chopper", wfm_chopper_indices[1]].data)

    for i in range(nframes):

        # Get frame parameters
        frame = data.meta["choppers"].value["frame", i]
        dist, tstart, tend = get_frame_properties(frame)

        # Find deltat for the min and max wavelengths:
        # - dt_max is equal to the time width of the WFM choppers windows
        dt_max = tend['chopper',
                      wfm_chopper_indices[0]] - tstart['chopper',
                                                       wfm_chopper_indices[0]]

        # - dt_min is found from the relation between lambda_min and lambda_max:
        # equation (3) in
        # https://www.sciencedirect.com/science/article/pii/S0168900220308640
        tmax = (dt_max / dz_wfm) * (pos_norm - z_wfm)
        tmin = tmax * (
            sc.norm(frame["position"]["chopper", wfm_chopper_indices[1]].data) /
            sc.norm(frame["position"]["chopper", wfm_chopper_indices[0]].data)
        ) - data.meta["pulse_length"] * (pos_norm / sc.norm(
            frame["position"]["chopper", wfm_chopper_indices[0]].data))
        dt_min = dz_wfm * tmin / (pos_norm - z_wfm)

        # Compute slopes
        origin_lambda_min = data.meta["pulse_t_0"] + data.meta["pulse_length"] - dt_min
        slopes_lambda_min = (dist - source_pos) / (tstart - origin_lambda_min)

        origin_lambda_max = data.meta["pulse_t_0"] + dt_max
        slopes_lambda_max = (dist - source_pos) / (tend - origin_lambda_max)

        # Find smallest of the lambda_min slopes
        slope_lambda_min = sc.min(slopes_lambda_min.data)
        # Find largest of the lambda max slopes
        slope_lambda_max = sc.max(slopes_lambda_max.data)

        # Keep a record of the dt resolutions for each frame
        frames["left_dt"]["frame", i] = dt_min
        frames["right_dt"]["frame", i] = dt_max

        # Compute line equation intercept y = slope*x + intercept
        intercept_lambda_min = source_pos - (slope_lambda_min * origin_lambda_min)
        intercept_lambda_max = source_pos - (slope_lambda_max * origin_lambda_max)

        # Frame edges for each pixel
        frames["left_edges"]["frame",
                             i] = (pos_norm - intercept_lambda_min) / slope_lambda_min
        frames["right_edges"]["frame",
                              i] = (pos_norm - intercept_lambda_max) / slope_lambda_max

        # Frame shifts
        frames["shifts"]["frame", i] = tstart["chopper", wfm_chopper_indices[1]]
        # sc.mean(
        #     sc.concatenate(tstart["chopper", 0:2], tend["chopper", 0:2],
        #                    "none"))

    frames["wfm_chopper_mid_point"] = sc.mean(
        sc.concatenate(
            data.meta["choppers"].value["position"]["chopper", wfm_chopper_indices[0]],
            data.meta["choppers"].value["position"]["chopper",
                                                    wfm_chopper_indices[1]], 'none'))

    # # Make figure if required
    # if plot:
    #     fig = _plot(data, frames)
    #     if isinstance(plot, str):
    #         fig.savefig(plot, bbox_inches='tight')

    return frames
