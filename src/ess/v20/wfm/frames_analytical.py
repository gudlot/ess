import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipp as sc


def _angular_frame_edge_to_time(angular_frequency, angle, phase, offset):
    """
    Convert an angle on a rotating chopper to a time point (in microseconds).
    """
    div = angular_frequency * (1.0 * sc.units.s)
    return (angle + phase) / div * (1.0e6 * sc.units.us) + offset


def _get_frame(frame_number, instrument, offset):
    """
    Get coordinates of a chopper frame opening in time and distance.
    """
    dist = sc.norm(instrument["distance"].data)
    tstart = _angular_frame_edge_to_time(
        instrument["angular_frequency"],
        instrument["frame_start"]["frame",
                                  frame_number], instrument["phase"], offset)
    tend = _angular_frame_edge_to_time(
        instrument["angular_frequency"], instrument["frame_end"]["frame",
                                                                 frame_number],
        instrument["phase"], offset)
    return dist, tstart, tend


def _get_source_pulse(pulse_length, offset):
    """
    Define source pulse start and end.
    By default, the source pulse is at distance 0.
    """
    x0 = (0.0 * sc.units.us) + offset
    x1 = pulse_length + offset
    y0 = 0.0 * sc.units.m
    return x0, x1, y0


def frames_analytical(instrument, plot=False, offset=None):
    """
    Compute analytical frame boundaries and shifts based on chopper
    parameters and detector pixel positions.
    A set of frame boundaries is returned for each pixel.
    The frame shifts are the same for all pixels.

    If `plot` is `True`, make a plot of the time-distance diagram.
    If `plot` is a string, make the plot and save it to a file whose name will
    be the same a `plot`.

    TODO: This currently ignores scattering paths, only the distance from
    source to pixel.
    For imaging, this is what we want, but for scattering techniques, we should
    use l1 + l2 in the future.
    """

    if offset is None:
        offset = 0.0 * sc.units.us

    # Compute distances for each pixel
    pos_norm = sc.norm(instrument["position"].data)

    # Define source pulse
    x0, x1, y0 = _get_source_pulse(instrument["pulse_length"], offset)

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

    for i in range(instrument.sizes["frame"]):

        # Compute slopes
        dist, xstart, xend = _get_frame(i, instrument, offset)
        slopes_min = (dist - y0).values / (xend - x0).values
        slopes_max = (dist - y0).values / (xstart - x1).values

        # Find largest of the minimum slopes
        imin = np.argmax(slopes_min)
        # Find smallest of the maximum slopes
        imax = np.argmin(slopes_max)

        # Compute line equation intercept y = slope*x + intercept
        intercept_min = y0 - (slopes_min[imin] * x0.value * y0.unit)
        intercept_max = y0 - (slopes_max[imax] * x1.value * y0.unit)

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

    # Make figure if required
    if plot:
        fig = _plot(instrument, frames, offset)
        if isinstance(plot, str):
            fig.savefig(plot, bbox_inches='tight')

    return frames


def _plot(instrument, frames, offset):
    """
    Plot the time-distance diagram that was used to compute the frame
    boundaries.
    """

    x0, x1, y0 = _get_source_pulse(instrument["pulse_length"], offset)

    # Find detector pixel furthest away from source
    pos_norm = sc.norm(instrument["position"].data)
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

    for i in range(instrument.sizes["frame"]):

        dist, xstart, xend = _get_frame(i, instrument, offset)

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
        ax.fill([x0.value, x1.value, left_edge.value, right_edge.value],
                [y0.value, y0.value, pos.value, pos.value],
                alpha=0.3,
                color=col)
        ax.plot([x0.value, right_edge.value], [y0.value, pos.value],
                color=col,
                lw=1)
        ax.plot([x1.value, left_edge.value], [y0.value, pos.value],
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
