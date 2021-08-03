# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipp as sc


def _angular_frame_edge_to_time(angular_frequency, angle, phase):
    """
    Convert an angle on a rotating chopper to a time point (in microseconds).
    """
    div = angular_frequency * (1.0 * sc.units.s)
    return (angle + phase) / div * (1.0e6 * sc.units.us)


def _get_frame(frame):
    """
    Get coordinates of a chopper frame opening in time and distance.
    """
    dist = sc.norm(frame["position"])
    tstart = _angular_frame_edge_to_time(frame["angular_frequency"],
                                         frame["opening_angles_open"],
                                         frame["phase"])
    tend = _angular_frame_edge_to_time(frame["angular_frequency"],
                                       frame["opening_angles_close"],
                                       frame["phase"])
    return dist, tstart, tend


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
    frames["left_edges"] = sc.zeros(
        dims=["frame"] + data.meta["position"].dims,
        shape=[nframes] + data.meta["position"].shape,
        unit=sc.units.us)
    frames["right_edges"] = sc.zeros_like(frames["left_edges"])
    frames["left_dt"] = sc.zeros_like(frames["left_edges"])
    frames["right_dt"] = sc.zeros_like(frames["left_edges"])
    frames["shifts"] = sc.zeros(dims=["frame"],
                                shape=[nframes],
                                unit=sc.units.us)

    # Identify the position of the WFM choppers in the array of choppers
    chopper_names = list(data.meta["choppers"].value["names"].values)
    wfm_chopper_indices = [
        chopper_names.index(name) for name in wfm_chopper_names
    ]

    # Order the WFM chopper indices from the chopper closest to the source to the
    # furthest away
    wfm_chopper_distances = [
        sc.norm(data.meta["choppers"].value["position"]["chopper",
                                                        ind].data).value
        for ind in wfm_chopper_indices
    ]
    wfm_chopper_indices = np.array(wfm_chopper_indices)[np.argsort(
        wfm_chopper_distances)]

    # Distance between WFM choppers
    dz_wfm = sc.norm(
        data.meta["choppers"].value["position"]["chopper",
                                                wfm_chopper_indices[1]].data -
        data.meta["choppers"].value["position"]["chopper",
                                                wfm_chopper_indices[0]].data)
    z_wfm = 0.5 * sc.norm(
        data.meta["choppers"].value["position"]["chopper",
                                                wfm_chopper_indices[0]].data +
        data.meta["choppers"].value["position"]["chopper",
                                                wfm_chopper_indices[1]].data)

    for i in range(nframes):

        # Get frame parameters
        frame = data.meta["choppers"].value["frame", i]
        dist, tstart, tend = _get_frame(frame)

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
            sc.norm(frame["position"]["chopper",
                                      wfm_chopper_indices[1]].data) /
            sc.norm(frame["position"]["chopper", wfm_chopper_indices[0]].data)
        ) - data.meta["pulse_length"] * (pos_norm / sc.norm(
            frame["position"]["chopper", wfm_chopper_indices[0]].data))
        dt_min = dz_wfm * tmin / (pos_norm - z_wfm)

        # Compute slopes
        origin_lambda_min = data.meta["pulse_t_0"] + data.meta[
            "pulse_length"] - dt_min
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
        intercept_lambda_min = source_pos - (slope_lambda_min *
                                             origin_lambda_min)
        intercept_lambda_max = source_pos - (slope_lambda_max *
                                             origin_lambda_max)

        # Frame edges for each pixel
        frames["left_edges"]["frame",
                             i] = (pos_norm -
                                   intercept_lambda_min) / slope_lambda_min
        frames["right_edges"]["frame",
                              i] = (pos_norm -
                                    intercept_lambda_max) / slope_lambda_max

        # Frame shifts
        frames["shifts"]["frame", i] = tstart["chopper",
                                              wfm_chopper_indices[1]]
        # sc.mean(
        #     sc.concatenate(tstart["chopper", 0:2], tend["chopper", 0:2],
        #                    "none"))

    # # Make figure if required
    # if plot:
    #     fig = _plot(data, frames)
    #     if isinstance(plot, str):
    #         fig.savefig(plot, bbox_inches='tight')

    return frames


def _plot(data, frames):
    """
    Plot the time-distance diagram that was used to compute the frame
    boundaries.
    """

    # x0, x1, y0 = _get_source_pulse(instrument["pulse_length"],
    #                                instrument["pulse_t_0"])

    # Find detector pixel furthest away from source
    pos_norm = sc.norm(data.meta["position"])
    source_pos = sc.norm(data.meta["source_position"])
    det_last = sc.max(pos_norm)
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    ax.grid(True, color='lightgray', linestyle="dotted")
    ax.set_axisbelow(True)
    psize = det_last.value / 50.0
    # rect = Rectangle((x0.value, y0.value),
    #                  x1.value - x0.value,
    #                  -psize,
    #                  lw=1,
    #                  fc='orange',
    #                  ec='k',
    #                  hatch="////",
    #                  zorder=10)
    ax.add_patch(
        Rectangle(
            (0, source_pos.value),
            (2.0 * data.meta["pulse_t_0"] + data.meta["pulse_length"]).value,
            -psize,
            lw=1,
            fc='lightgrey',
            ec='k',
            zorder=10))
    ax.add_patch(
        Rectangle((data.meta["pulse_t_0"].value, source_pos.value),
                  data.meta["pulse_length"].value,
                  -psize,
                  lw=1,
                  fc='grey',
                  ec='k',
                  zorder=11))
    ax.text(data.meta["pulse_t_0"].value,
            -psize,
            "Source pulse ({} {})".format(data.meta["pulse_length"].value,
                                          data.meta["pulse_length"].unit),
            ha="left",
            va="top",
            fontsize=6)

    for i in range(
            data.meta["choppers"].value["opening_angles_open"].sizes["frame"]):

        dist, xstart, xend = _get_frame(data.meta["choppers"].value["frame",
                                                                    i])
        # dist, xstart, xend = _get_frame(i, instrument)

        for j in range(data.meta["choppers"].value["opening_angles_open"].
                       sizes["chopper"]):
            ax.plot([xstart["chopper", j].value, xend["chopper", j].value],
                    [dist["chopper", j].value] * 2,
                    color="C{}".format(i))
            if i == data.meta["choppers"].value["opening_angles_open"].sizes[
                    "frame"] - 1:
                ax.text((2.0 * xend["chopper", j].data -
                         xstart["chopper", j]).value,
                        dist["chopper", j].value,
                        data.meta["choppers"].value["names"]["chopper",
                                                             j].value,
                        ha="left",
                        va="center")

        col = "C{}".format(i)
        # left_edge = frames["left_edges"]["frame", i]
        # right_edge = frames["right_edges"]["frame", i]
        frame = frames["frame", i]
        # right_edge = frames["right_edges"]["frame", i]
        pos = pos_norm.copy()
        for dim in data.meta["position"].dims:
            # left_edge = left_edge[dim, 0]
            # right_edge = right_edge[dim, 0]
            frame = frame[dim, 0]
            pos = pos[dim, 0]
        ax.fill([(data.meta["pulse_t_0"] + frame["right_dt"]).value,
                 (data.meta["pulse_t_0"] + data.meta["pulse_length"] -
                  frame["left_dt"]).value,
                 (frame["left_edges"] + (0.5 * frame["left_dt"].data)).value,
                 (frame["right_edges"] -
                  (0.5 * frame["right_dt"].data)).value],
                [source_pos.value, source_pos.value, pos.value, pos.value],
                alpha=0.3,
                color=col)
        ax.fill([
            data.meta["pulse_t_0"].value,
            (data.meta["pulse_t_0"] + frame["right_dt"]).value,
            (frame["right_edges"] + (0.5 * frame["right_dt"].data)).value,
            (frame["right_edges"] - (0.5 * frame["right_dt"].data)).value
        ], [source_pos.value, source_pos.value, pos.value, pos.value],
                alpha=0.15,
                color=col)
        ax.fill([(data.meta["pulse_t_0"] + data.meta["pulse_length"] -
                  frame["left_dt"]).value,
                 (data.meta["pulse_t_0"] + data.meta["pulse_length"]).value,
                 (frame["left_edges"] + (0.5 * frame["left_dt"].data)).value,
                 (frame["left_edges"] - (0.5 * frame["left_dt"].data)).value],
                [source_pos.value, source_pos.value, pos.value, pos.value],
                alpha=0.15,
                color=col)

        # Minimum wavelength
        ax.plot([(data.meta["pulse_t_0"] + data.meta["pulse_length"]).value,
                 (frame["left_edges"] + (0.5 * frame["left_dt"].data)).value],
                [source_pos.value, pos.value],
                color=col,
                lw=1)
        ax.plot([(data.meta["pulse_t_0"] + data.meta["pulse_length"] -
                  frame['left_dt']).value,
                 (frame["left_edges"] - (0.5 * frame["left_dt"].data)).value],
                [source_pos.value, pos.value],
                color=col,
                lw=1)
        # Maximum wavelength
        ax.plot([
            data.meta["pulse_t_0"].value,
            (frame["right_edges"] - (0.5 * frame["right_dt"].data)).value
        ], [source_pos.value, pos.value],
                color=col,
                lw=1)
        ax.plot([(data.meta["pulse_t_0"] + frame["right_dt"]).value,
                 (frame["right_edges"] +
                  (0.5 * frame["right_dt"].data)).value],
                [source_pos.value, pos.value],
                color=col,
                lw=1)

        # ax.plot([(instrument["pulse_t_0"] + instrument["pulse_length"]).value,
        #          left_edge.value], [source_pos.value, pos.value],
        #         color=col,
        #         lw=1)
        # ax.text(0.5 * (left_edge + right_edge).value,
        #         pos.value,
        #         "Frame {}".format(i + 1),
        #         ha="center",
        #         va="top")

    ax.plot([0, sc.max(frames["right_edges"].data).value],
            [det_last.value] * 2,
            lw=3,
            color='grey')
    ax.text(0.0, det_last.value, "Detector", va="bottom", ha="left")
    ax.set_xlabel("Time [microseconds]")
    ax.set_ylabel("Distance [m]")

    return fig
