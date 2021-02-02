import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipp as sc


def frames_analytical(instrument=None, plot=False, offset=None):


    if offset is None:
        offset = 0.0 * sc.units.us
    # info = instrument.coords
    # choppers = instrument["choppers"]

    # Find key of detector furthest away from source
    # imax = np.argmax(list(info["detector_positions"].values()))
    # det_last = list(info["detector_positions"].keys())[imax]
    # det_last["position"]
    # det_last = sc.max(sc.geometry.z(info["position"]))
    pos_norm = sc.norm(instrument["position"].data)
    det_last = sc.max(pos_norm)

    # Seconds to microseconds
    s_to_us = (1.0e6 * sc.units.us)# / (1.0 * sc.units.s)

    # Define and draw source pulse
    x0 = (0.0 * sc.units.us) + offset
    x1 = instrument["pulse_length"] + offset
    y0 = 0.0 * sc.units.m
    y1 = 0.0 * sc.units.m

    # Make figure
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        ax.grid(True, color='lightgray', linestyle="dotted")
        ax.set_axisbelow(True)

        # # Plot the chopper openings
        # for key, chopper in instrument.items():
        #     dist = sc.norm(chopper.attrs["distance"]).value
        #     for i in range(0, chopper.shape[0], 2):
        #         t1 = (chopper["openings", i] +
        #               chopper.attrs["phase"]) / chopper.attrs["angular_frequency"] / (1.0 * sc.units.s) * s_to_us + offset
        #         t2 = (chopper["openings", i + 1] +
        #               chopper.attrs["phase"]) / chopper.attrs["angular_frequency"] / (1.0 * sc.units.s) * s_to_us + offset
        #         ax.plot([t1.value, t2.value], [dist, dist], color="C{}".format(i // 2))
        #     ax.text(t2.value + (t2.value - t1.value),
        #             dist,
        #             key,
        #             ha="left",
        #             va="center")

        # psize = info["detector_positions"][det_last] / 50.0
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
    frames["left_edges"] = sc.zeros(dims=["frame"] + instrument["position"].dims, shape=[instrument.sizes["frame"]] + instrument["position"].shape,
        unit=sc.units.us)
    frames["right_edges"] = frames["left_edges"].copy()
    frames["shifts"] = frames["left_edges"].copy()
    # return frames
    # for det in info["detector_positions"]:
    #     frames[det] = {"left_edges": [], "right_edges": [], "shifts": []}

    for i in range(instrument.sizes["frame"]):

        # Compute slopes
        dist = sc.norm(instrument["distance"].data)
        div = instrument["angular_frequency"] * (1.0 * sc.units.s)
        xstart = (instrument["frame_start"]["frame", i] + instrument["phase"]) / div *   s_to_us + offset
        xend = (instrument["frame_end"]["frame", i] + instrument["phase"]) / div * s_to_us + offset

        slopes_min = (dist - y0).values / (xend - x0).values
        slopes_max = (dist - y1).values / (xstart - x1).values

        # Find largest of the minimum slopes
        imin = np.argmax(slopes_min)
        # Find smallest of the maximum slopes
        imax = np.argmin(slopes_max)

        if plot:
            for j in range(instrument.sizes["chopper"]):
                ax.plot([xstart["chopper", j].value, xend["chopper", j].value],
                    [dist["chopper", j].value]*2, color="C{}".format(i))
                if i == instrument.sizes["frame"]-1:
                    ax.text((2.0*xend["chopper", j].data - xstart["chopper", j]).value,
                            dist["chopper", j].value,
                            instrument["choppers"]["chopper", j].value,
                            ha="left",
                            va="center")

        # return
        # # Find the minimum and maximum slopes that are allowed through each frame
        # slope_min = 1.0e30
        # slope_max = -1.0e30
        # for key, chopper in instrument.items():

        #     # xmin = (chopper.openings[i * 2] +
        #     #         ch.phase) / ch.omega * microseconds + offset
        #     xmin = (chopper["frame", i]["state", 0] +
        #               chopper.attrs["phase"]) / chopper.attrs["angular_frequency"] / (1.0 * sc.units.s) * s_to_us + offset
        #     xmax = (chopper["frame", i]["state", 1] +
        #               chopper.attrs["phase"]) / chopper.attrs["angular_frequency"] / (1.0 * sc.units.s) * s_to_us + offset

        #     dist = sc.norm(chopper.attrs["distance"]).value

        #     if plot:
        #         ax.plot([xmin.value, xmax.value], [dist, dist], color="C{}".format(i // 2))
        #         if i == info["nframes"].value - 1:
        #             ax.text(xmax.value + (xmax.value - xmin.value),
        #                     dist,
        #                     key,
        #                     ha="left",
        #                     va="center")

        #     # Compute slopes
        #     slope1 = (dist - y1) / (xmin.value - x1)
        #     slope2 = (dist - y0) / (xmax.value - x0)

        #     if slope_min > slope1:
        #         x2 = xmin.value
        #         y2 = dist
        #         slope_min = slope1
        #     if slope_max < slope2:
        #         x3 = xmax.value
        #         y3 = dist
        #         slope_max = slope2

        # Compute line equation intercept y = slope*x + intercept
        # a1 = (dist["chopper", imax] - y0) / (xmax["chopper", imax] - x0)
        # a2 = (y2 - y1) / (x2 - x1)
        intercept_min = y0 - (slopes_min[imin] * x0.value * y0.unit)
        intercept_max = y1 - (slopes_max[imax] * x1.value * y1.unit)

        # ax.plot([x0.value, xend["chopper", imin].value], [y0.value, dist["chopper", imin].value])
        # ax.plot([x1.value, xstart["chopper", imax].value], [y0.value, dist["chopper", imax].value])

        # return

        # # # Compute frame shifts from WFM chopper openings centre of mass
        # # s0 = (choppers["WFM1"].openings[i * 2] + choppers["WFM1"].phase
        # #       ) / choppers["WFM1"].omega * microseconds + offset
        # # s1 = (choppers["WFM1"].openings[i * 2 + 1] + choppers["WFM1"].phase
        # #       ) / choppers["WFM1"].omega * microseconds + offset
        # # s2 = (choppers["WFM2"].openings[i * 2] + choppers["WFM2"].phase
        # #       ) / choppers["WFM2"].omega * microseconds + offset
        # # s3 = (choppers["WFM2"].openings[i * 2 + 1] + choppers["WFM2"].phase
        # #       ) / choppers["WFM2"].omega * microseconds + offset
        # # shift = -0.25 * (s0 + s1 + s2 + s3)

        # shift1 = (instrument["WFM1"]["frame", i] +
        #               chopper.attrs["phase"]) / chopper.attrs["angular_frequency"] / (1.0 * sc.units.s) * s_to_us + offset
        # shift2 = (instrument["WFM2"]["frame", i] +
        #       chopper.attrs["phase"]) / chopper.attrs["angular_frequency"] / (1.0 * sc.units.s) * s_to_us + offset
        # shift = sc.mean(sc.concatenate(shift1, shift2, "none"))
        # print(shift)


        frames["right_edges"]["frame", i] = ((pos_norm - intercept_min).values / slopes_min[imin]) * sc.units.us
        frames["left_edges"]["frame", i] = ((pos_norm - intercept_max).values / slopes_max[imax]) * sc.units.us

        frames["shifts"]["frame", i] = sc.mean(sc.concatenate(
            xstart["chopper", 0:2], xend["chopper", 0:2], "none"))

        # print(frames["left_edges"]["frame", i])
        # print(frames["right_edges"]["frame", i])

        # for det in info["detector_positions"]:
        #     y4 = info["detector_positions"][det]
        #     y5 = info["detector_positions"][det]

        #     # This is the frame boundaries
        #     x5 = (y5 - b1) / a1
        #     x4 = (y4 - b2) / a2
        #     frames[det]["left_edges"].append(x4)
        #     frames[det]["right_edges"].append(x5)
        #     frames[det]["shifts"].append(shift)

        if plot:
            col = "C{}".format(i)
            left_edge = frames["left_edges"]["frame", i]
            right_edge = frames["right_edges"]["frame", i]
            pos = pos_norm.copy()
            for dim in instrument["position"].dims:
                left_edge = left_edge[dim, 0]
                right_edge = right_edge[dim, 0]
                pos = pos[dim, 0]
            # print(left_edge)
            # print(right_edge)
            # print(pos)
            ax.fill([
                x0.value, x1.value, left_edge.value, right_edge.value
            ], [
                y0.value, y1.value, pos.value,
                pos.value
            ],
                    alpha=0.3,
                    color=col)
            ax.plot([x0.value, right_edge.value],
                    [y0.value, pos.value],
                    color=col,
                    lw=1)
            ax.plot([x1.value, left_edge.value],
                    [y1.value, pos.value],
                    color=col,
                    lw=1)
            ax.text(0.5 * (left_edge + right_edge).value,
                    pos.value,
                    "Frame {}".format(i + 1),
                    ha="center",
                    va="top")

    if plot:
        ax.plot([0, sc.max(frames["right_edges"].data).value],
                [det_last.value]*2,
                lw=3,
                color='grey')
        ax.text(0.0, det_last.value, "Detector", va="bottom", ha="left")
        ax.set_xlabel("Time [microseconds]")
        ax.set_ylabel("Distance [m]")

    return frames
