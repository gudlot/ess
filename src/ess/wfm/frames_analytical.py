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


def _get_frame(frame_number, instrument):
    """
    Get coordinates of a chopper frame opening in time and distance.
    """
    dist = sc.norm(instrument["distance"].data)
    tstart = _angular_frame_edge_to_time(
        instrument["angular_frequency"],
        instrument["frame_start"]["frame", frame_number], instrument["phase"])
    tend = _angular_frame_edge_to_time(
        instrument["angular_frequency"],
        instrument["frame_end"]["frame", frame_number], instrument["phase"])
    return dist, tstart, tend


# def _get_source_pulse(pulse_length, t_0):
#     """
#     Define source pulse start and end.
#     By default, the source pulse is at distance 0.
#     """
#     x0 = (0.0 * sc.units.us) + t_0
#     x1 = pulse_length + t_0
#     y0 = 0.0 * sc.units.m
#     return x0, x1, y0


def frames_analytical(instrument, plot=False):
    """
    Compute analytical frame boundaries and shifts based on chopper
    parameters and detector pixel positions.
    A set of frame boundaries is returned for each pixel.
    The frame shifts are the same for all pixels.
    See https://www.sciencedirect.com/science/article/pii/S0168900220308640
    for a description of the procedure.

    If `plot` is `True`, make a plot of the time-distance diagram.
    If `plot` is a string, make the plot and save it to a file whose name will
    be the same a `plot`.

    TODO: This currently ignores scattering paths, only the distance from
    source to pixel.
    For imaging, this is what we want, but for scattering techniques, we should
    use l1 + l2 in the future.
    """

    # if offset is None:
    #     offset = 0.0 * sc.units.us
    # neutron_mass = sc.scalar(1.67492749804e-27, unit='kg')
    # planck_constant = sc.scalar(6.62607004e-34, unit='m*m*kg/s')
    # alpha = neutron_mass / planck_constant

    # Compute distances for each pixel
    pos_norm = sc.norm(instrument["position"].data)
    source_pos = sc.norm(instrument["source_position"])

    # # Define source pulse
    # x0, x1, y0 = _get_source_pulse(instrument["pulse_length"],
    #                                instrument["pulse_t_0"])

    # Now find frame boundaries
    frames = sc.Dataset()
    frames["left_edges"] = sc.zeros(
        dims=["frame"] + instrument["position"].dims,
        shape=[instrument.sizes["frame"]] + instrument["position"].shape,
        unit=sc.units.us)
    frames["right_edges"] = frames["left_edges"].copy()
    frames["shifts"] = sc.zeros(dims=["frame"],
                                shape=[instrument.sizes["frame"]],
                                unit=sc.units.us)

    # Distance between WFM choppers
    dz_wfm = sc.norm(instrument["distance"]["chopper", 1].data -
                     instrument["distance"]["chopper", 0].data)
    z_wfm = 0.5 * sc.norm(instrument["distance"]["chopper", 0].data +
                          instrument["distance"]["chopper", 1].data)

    for i in range(instrument.sizes["frame"]):

        # Get frame parameters
        dist, tstart, tend = _get_frame(i, instrument)
        # print(dist, tstart, tend)

        # Find deltat for the min and max wavelengths:
        # - dt_max is equal to the time width of the WFM choppers windows
        dt_max = tend['chopper', 0] - tstart['chopper', 0]

        # - dt_min is found from the relation between lambda_min and lambda_max
        tmax = (dt_max / dz_wfm) * (pos_norm - z_wfm)
        # lambda_min = (tmax * sc.norm(instrument["WFMC2"].distance.data) /
        #               pos_norm - instrument["pulse_length"]) / (
        #                   alpha * sc.norm(instrument["WFMC1"].distance.data))
        # print(tmax * (sc.norm(instrument["distance"]["chopper", 1].data) /
        #               sc.norm(instrument["distance"]["chopper", 0].data)))
        tmin = tmax * (sc.norm(instrument["distance"]["chopper", 1].data) /
                       sc.norm(instrument["distance"]["chopper", 0].data)
                       ) - instrument["pulse_length"] * (pos_norm / sc.norm(
                           instrument["distance"]["chopper", 0].data))
        dt_min = dz_wfm * tmin / (pos_norm - z_wfm)

        # # For left edge of frame, find rightmost chopper leading edge
        # tstart_max = sc.max(tstart)
        # # For right edge of frame, find leftmost chopper closing edge
        # tend_min = sc.min(tend)

        # Compute slopes
        origin_lambda_min = instrument["pulse_t_0"] + instrument[
            "pulse_length"] - dt_min
        slopes_lambda_min = (dist - source_pos) / (tstart - origin_lambda_min)

        origin_lambda_max = instrument["pulse_t_0"] + dt_max
        slopes_lambda_max = (dist - source_pos) / (tend - origin_lambda_max)

        # Find largest of the minimum slopes
        slope_lambda_min = sc.max(slopes_lambda_min.data)
        # Find smallest of the maximum slopes
        slope_lambda_max = sc.min(slopes_lambda_max.data)

        # Compute line equation intercept y = slope*x + intercept
        intercept_lambda_min = source_pos - (slope_lambda_min *
                                             origin_lambda_min)
        intercept_lambda_max = source_pos - (slope_lambda_max *
                                             origin_lambda_max)

        # print(intercept_lambda_min, pos_norm)
        # Frame edges for each pixel
        frames["left_edges"]["frame",
                             i] = (pos_norm -
                                   intercept_lambda_min) / slope_lambda_min
        frames["right_edges"]["frame",
                              i] = (pos_norm -
                                    intercept_lambda_max) / slope_lambda_max

        # def make_edge(dims, x):
        #     kwargs = {'unit': sc.units.us}
        #     if dims:
        #         return sc.array(dims=dims, values=x, **kwargs)
        #     else:
        #         return sc.scalar(value=x, **kwargs)

        # # Frame edges for each pixel
        # frames["right_edges"]["frame", i] = make_edge(
        #     pos_norm.dims,
        #     (pos_norm - intercept_min).values / slopes_min[imin])
        # frames["left_edges"]["frame", i] = make_edge(
        #     pos_norm.dims,
        #     (pos_norm - intercept_max).values / slopes_max[imax])

        # # # Frame edges for each pixel
        # frames["right_edges"]["frame", i] = sc.array(
        #     dims=pos_norm.dims,
        #     values=((pos_norm - intercept_min).values / slopes_min[imin]),
        #     unit=sc.units.us)
        # frames["left_edges"]["frame", i] = sc.array(
        #     dims=pos_norm.dims,
        #     values=((pos_norm - intercept_max).values / slopes_max[imax]),
        #     unit=sc.units.us)
        # Frame shifts
        frames["shifts"]["frame", i] = sc.mean(
            sc.concatenate(tstart["chopper", 0:2], tend["chopper", 0:2],
                           "none"))

    print(frames)

    # Make figure if required
    if plot:
        fig = _plot(instrument, frames)
        if isinstance(plot, str):
            fig.savefig(plot, bbox_inches='tight')

    return frames


def _plot(instrument, frames):
    """
    Plot the time-distance diagram that was used to compute the frame
    boundaries.
    """

    # x0, x1, y0 = _get_source_pulse(instrument["pulse_length"],
    #                                instrument["pulse_t_0"])

    # Find detector pixel furthest away from source
    pos_norm = sc.norm(instrument["position"].data)
    source_pos = sc.norm(instrument["source_position"])
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
        Rectangle((0, source_pos.value),
                  (2.0 * instrument["pulse_t_0"].data +
                   instrument["pulse_length"]).value,
                  -psize,
                  lw=1,
                  fc='lightgrey',
                  ec='k',
                  zorder=10))
    ax.add_patch(
        Rectangle((instrument["pulse_t_0"].value, source_pos.value),
                  instrument["pulse_length"].value,
                  -psize,
                  lw=1,
                  fc='grey',
                  ec='k',
                  zorder=11))
    ax.text(instrument["pulse_t_0"].value,
            -psize,
            "Source pulse ({} {})".format(instrument["pulse_length"].value,
                                          instrument["pulse_length"].unit),
            ha="left",
            va="top",
            fontsize=6)

    for i in range(instrument.sizes["frame"]):

        dist, xstart, xend = _get_frame(i, instrument)

        for j in range(instrument.sizes["chopper"]):
            ax.plot([xstart["chopper", j].value, xend["chopper", j].value],
                    [dist["chopper", j].value] * 2,
                    color="C{}".format(i))
            if i == instrument.sizes["frame"] - 1:
                ax.text((2.0 * xend["chopper", j].data -
                         xstart["chopper", j]).value,
                        dist["chopper", j].value,
                        instrument["choppers"]["chopper", j].value,
                        ha="left",
                        va="center")

        col = "C{}".format(i)
        left_edge = frames["left_edges"]["frame", i]
        right_edge = frames["right_edges"]["frame", i]
        pos = pos_norm.copy()
        for dim in instrument["position"].dims:
            left_edge = left_edge[dim, 0]
            right_edge = right_edge[dim, 0]
            pos = pos[dim, 0]
        ax.fill([
            instrument["pulse_t_0"].value,
            (instrument["pulse_t_0"] + instrument["pulse_length"]).value,
            left_edge.value, right_edge.value
        ], [source_pos.value, source_pos.value, pos.value, pos.value],
                alpha=0.3,
                color=col)
        ax.plot([instrument["pulse_t_0"].value, right_edge.value],
                [source_pos.value, pos.value],
                color=col,
                lw=1)
        ax.plot([(instrument["pulse_t_0"] + instrument["pulse_length"]).value,
                 left_edge.value], [source_pos.value, pos.value],
                color=col,
                lw=1)
        ax.text(0.5 * (left_edge + right_edge).value,
                pos.value,
                "Frame {}".format(i + 1),
                ha="center",
                va="top")

    ax.plot([0, sc.max(frames["right_edges"].data).value],
            [det_last.value] * 2,
            lw=3,
            color='grey')
    ax.text(0.0, det_last.value, "Detector", va="bottom", ha="left")
    ax.set_xlabel("Time [microseconds]")
    ax.set_ylabel("Distance [m]")

    return fig
