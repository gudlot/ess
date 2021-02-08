import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipp as sc


def frames_analytical(instrument=None, plot=False, offset=None):
    """
    Compute analytical frame boundaries and shifts based on chopper
    parameters and detector pixel positions.
    A set of frame boundaries is returned for each pixel.
    The frame shifts are the same for all pixels.

    TODO: This currently ignores scattering paths, only the distance from
    source to pixel. We should use l1 + l2 in the future.
    """

    if offset is None:
        offset = 0.0 * sc.units.us

    # Compute distances for each pixel
    pos_norm = sc.norm(instrument["position"].data)

    # Seconds to microseconds
    s_to_us = (1.0e6 * sc.units.us)

    # Define and draw source pulse
    x0 = (0.0 * sc.units.us) + offset
    x1 = instrument["pulse_length"] + offset
    y0 = 0.0 * sc.units.m
    y1 = 0.0 * sc.units.m

    # Make figure
    if plot:
        # Find detector pixel furthest away from source
        det_last = sc.max(pos_norm)
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        ax.grid(True, color='lightgray', linestyle="dotted")
        ax.set_axisbelow(True)
        psize = det_last.value / 50.0
        rect = Rectangle((x0.value, y0.value),
                         x1.value - x0.value,
                         -psize,
                         lw=1,
                         fc='orange',
                         ec='k',
                         hatch="////",
                         zorder=10)
        ax.add_patch(rect)
        ax.text(x0.value,
                -psize,
                "Source pulse (2.86 ms)",
                ha="left",
                va="top",
                fontsize=6)

    # Now find frame boundaries and draw frames
    frames = sc.Dataset()
    frames["left_edges"] = sc.zeros(
        dims=["frame"] + instrument["position"].dims,
        shape=[instrument.sizes["frame"]] + instrument["position"].shape,
        unit=sc.units.us)
    frames["right_edges"] = frames["left_edges"].copy()
    frames["shifts"] = sc.zeros(dims=["frame"],
                                shape=[instrument.sizes["frame"]],
                                unit=sc.units.us)

    for i in range(instrument.sizes["frame"]):

        # Compute slopes
        dist = sc.norm(instrument["distance"].data)
        div = instrument["angular_frequency"] * (1.0 * sc.units.s)
        xstart = (instrument["frame_start"]["frame", i] +
                  instrument["phase"]) / div * s_to_us + offset
        xend = (instrument["frame_end"]["frame", i] +
                instrument["phase"]) / div * s_to_us + offset

        slopes_min = (dist - y0).values / (xend - x0).values
        slopes_max = (dist - y1).values / (xstart - x1).values

        # Find largest of the minimum slopes
        imin = np.argmax(slopes_min)
        # Find smallest of the maximum slopes
        imax = np.argmin(slopes_max)

        if plot:
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

        # Compute line equation intercept y = slope*x + intercept
        intercept_min = y0 - (slopes_min[imin] * x0.value * y0.unit)
        intercept_max = y1 - (slopes_max[imax] * x1.value * y1.unit)

        # Frame edges for each pixel
        frames["right_edges"]["frame", i] = sc.array(
            dims=pos_norm.dims,
            values=((pos_norm - intercept_min).values / slopes_min[imin]),
            unit=sc.units.us)
        frames["left_edges"]["frame", i] = sc.array(
            dims=pos_norm.dims,
            values=((pos_norm - intercept_max).values / slopes_max[imax]),
            unit=sc.units.us)
        # Frame shifts
        frames["shifts"]["frame", i] = sc.mean(
            sc.concatenate(xstart["chopper", 0:2], xend["chopper", 0:2],
                           "none"))

        if plot:
            col = "C{}".format(i)
            left_edge = frames["left_edges"]["frame", i]
            right_edge = frames["right_edges"]["frame", i]
            pos = pos_norm.copy()
            for dim in instrument["position"].dims:
                left_edge = left_edge[dim, 0]
                right_edge = right_edge[dim, 0]
                pos = pos[dim, 0]
            ax.fill([x0.value, x1.value, left_edge.value, right_edge.value],
                    [y0.value, y1.value, pos.value, pos.value],
                    alpha=0.3,
                    color=col)
            ax.plot([x0.value, right_edge.value], [y0.value, pos.value],
                    color=col,
                    lw=1)
            ax.plot([x1.value, left_edge.value], [y1.value, pos.value],
                    color=col,
                    lw=1)
            ax.text(0.5 * (left_edge + right_edge).value,
                    pos.value,
                    "Frame {}".format(i + 1),
                    ha="center",
                    va="top")

    if plot:
        ax.plot([0, sc.max(frames["right_edges"].data).value],
                [det_last.value] * 2,
                lw=3,
                color='grey')
        ax.text(0.0, det_last.value, "Detector", va="bottom", ha="left")
        ax.set_xlabel("Time [microseconds]")
        ax.set_ylabel("Distance [m]")

    return frames
