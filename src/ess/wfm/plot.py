# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipp as sc
from .wfm import get_frames
from .tools import frame_opening_and_closing_times


def time_distance_diagram(data: sc.DataArray, **kwargs) -> plt.Figure:
    """
    Plot the time-distance diagram for a WFM beamline.
    This internally calls the `get_frames` method which is used to compute the
    frame properties for stitching.
    """

    frames = get_frames(data, **kwargs)

    # Find detector pixel furthest away from source
    pos_norm = sc.norm(data.meta["position"])
    source_pos = sc.norm(data.meta["source_position"])
    det_last = sc.max(pos_norm)
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    ax.grid(True, color='lightgray', linestyle="dotted")
    ax.set_axisbelow(True)
    psize = det_last.value / 50.0

    ax.add_patch(
        Rectangle((0, source_pos.value),
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

    for i in range(data.meta["choppers"].value["opening_angles_open"].sizes["frame"]):

        xstart, xend = frame_opening_and_closing_times(
            data.meta["choppers"].value["frame", i])

        yframe = sc.norm(data.meta["choppers"].value["frame", i]["position"])

        for j in range(
                data.meta["choppers"].value["opening_angles_open"].sizes["chopper"]):
            ax.plot([xstart["chopper", j].value, xend["chopper", j].value],
                    [yframe["chopper", j].value] * 2,
                    color="C{}".format(i))
            if i == data.meta["choppers"].value["opening_angles_open"].sizes[
                    "frame"] - 1:
                ax.text((2.0 * xend["chopper", j] - xstart["chopper", j]).value,
                        yframe["chopper", j].value,
                        data.meta["choppers"].value["names"]["chopper", j].value,
                        ha="left",
                        va="center")

        col = "C{}".format(i)
        frame = frames["frame", i]
        pos = pos_norm.copy()
        for dim in data.meta["position"].dims:
            frame = frame[dim, 0]
            pos = pos[dim, 0]

        # Minimum wavelength
        lambda_min = np.array([[
            (data.meta["pulse_t_0"] + data.meta["pulse_length"] -
             frame['left_dt']).value, source_pos.value
        ], [(data.meta["pulse_t_0"] + data.meta["pulse_length"]).value,
            source_pos.value],
                               [(frame["left_edges"] + frame["left_dt"]).value,
                                pos.value], [frame["left_edges"].value, pos.value]])

        # Maximum wavelength
        lambda_max = np.array([[data.meta["pulse_t_0"].value, source_pos.value],
                               [(data.meta["pulse_t_0"] + frame['right_dt']).value,
                                source_pos.value],
                               [frame["right_edges"].value, pos.value],
                               [(frame["right_edges"] - frame["right_dt"]).value,
                                pos.value]])

        ax.plot(np.concatenate((lambda_min[:, 0], lambda_min[0:1, 0])),
                np.concatenate((lambda_min[:, 1], lambda_min[0:1, 1])),
                color=col,
                lw=1)

        ax.plot(np.concatenate((lambda_max[:, 0], lambda_max[0:1, 0])),
                np.concatenate((lambda_max[:, 1], lambda_max[0:1, 1])),
                color=col,
                lw=1)

        ax.fill(
            [lambda_max[0, 0], lambda_max[-1, 0], lambda_min[2, 0], lambda_min[1, 0]],
            [lambda_max[0, 1], lambda_max[-1, 1], lambda_min[2, 1], lambda_min[1, 1]],
            alpha=0.3,
            color=col,
            zorder=-5)

        ax.fill(lambda_min[:, 0], lambda_min[:, 1], color='w', zorder=-4)
        ax.fill(lambda_max[:, 0], lambda_max[:, 1], color='w', zorder=-4)

        ax.text(sc.mean(
            sc.concatenate(frame["left_edges"], frame["right_edges"], 'none')).value,
                det_last.value,
                "Frame {}".format(i + 1),
                ha="center",
                va="top")

    ax.plot([0, sc.max(frames["right_edges"].data).value], [det_last.value] * 2,
            lw=3,
            color='grey')
    ax.text(0.0, det_last.value, "Detector", va="bottom", ha="left")
    ax.set_xlabel("Time [microseconds]")
    ax.set_ylabel("Distance [m]")

    return fig
