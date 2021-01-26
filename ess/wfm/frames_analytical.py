import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def frames_analytical(instrument=None, plot=False, offset=0.0):

    info = instrument["info"]
    choppers = instrument["choppers"]

    # Find key of detector furthest away from source
    imax = np.argmax(list(info["detector_positions"].values()))
    det_last = list(info["detector_positions"].keys())[imax]

    # Seconds to microseconds
    microseconds = 1.0e6

    # Define and draw source pulse
    x0 = 0.0 + offset
    x1 = (info["pulse_length"] * microseconds) + offset
    y0 = 0.0
    y1 = 0.0

    # Make figure
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        ax.grid(True, color='lightgray', linestyle="dotted")
        ax.set_axisbelow(True)

        # Plot the chopper openings
        for key, ch in choppers.items():
            dist = [ch.distance, ch.distance]
            for i in range(0, len(ch.openings), 2):
                t1 = (ch.openings[i] +
                      ch.phase) / ch.omega * microseconds + offset
                t2 = (ch.openings[i + 1] +
                      ch.phase) / ch.omega * microseconds + offset
                ax.plot([t1, t2], dist, color="C{}".format(i // 2))
            ax.text(t2 + (t2 - t1),
                    ch.distance,
                    ch.name,
                    ha="left",
                    va="center")

        psize = info["detector_positions"][det_last] / 50.0
        rect = Rectangle((x0, y0),
                         x1 - x0,
                         -psize,
                         lw=1,
                         fc='orange',
                         ec='k',
                         hatch="////",
                         zorder=10)
        ax.add_patch(rect)
        ax.text(x0,
                -psize,
                "Source pulse (2.86 ms)",
                ha="left",
                va="top",
                fontsize=6)

    # Now find frame boundaries and draw frames
    frames = {}
    for det in info["detector_positions"]:
        frames[det] = {"left_edges": [], "right_edges": [], "shifts": []}

    for i in range(info["nframes"]):

        # Find the minimum and maximum slopes that are allowed through each frame
        slope_min = 1.0e30
        slope_max = -1.0e30
        for key, ch in choppers.items():

            # For now, ignore Wavelength band double chopper
            if len(ch.openings) == info["nframes"] * 2:

                xmin = (ch.openings[i * 2] +
                        ch.phase) / ch.omega * microseconds + offset
                xmax = (ch.openings[i * 2 + 1] +
                        ch.phase) / ch.omega * microseconds + offset
                slope1 = (ch.distance - y1) / (xmin - x1)
                slope2 = (ch.distance - y0) / (xmax - x0)

                if slope_min > slope1:
                    x2 = xmin
                    y2 = ch.distance
                    slope_min = slope1
                if slope_max < slope2:
                    x3 = xmax
                    y3 = ch.distance
                    slope_max = slope2

        # Compute line equation parameters y = a*x + b
        a1 = (y3 - y0) / (x3 - x0)
        a2 = (y2 - y1) / (x2 - x1)
        b1 = y0 - a1 * x0
        b2 = y1 - a2 * x1

        # Compute frame shifts from WFM chopper openings centre of mass
        s0 = (choppers["WFM1"].openings[i * 2] + choppers["WFM1"].phase
              ) / choppers["WFM1"].omega * microseconds + offset
        s1 = (choppers["WFM1"].openings[i * 2 + 1] + choppers["WFM1"].phase
              ) / choppers["WFM1"].omega * microseconds + offset
        s2 = (choppers["WFM2"].openings[i * 2] + choppers["WFM2"].phase
              ) / choppers["WFM2"].omega * microseconds + offset
        s3 = (choppers["WFM2"].openings[i * 2 + 1] + choppers["WFM2"].phase
              ) / choppers["WFM2"].omega * microseconds + offset
        shift = -0.25 * (s0 + s1 + s2 + s3)

        for det in info["detector_positions"]:
            y4 = info["detector_positions"][det]
            y5 = info["detector_positions"][det]

            # This is the frame boundaries
            x5 = (y5 - b1) / a1
            x4 = (y4 - b2) / a2
            frames[det]["left_edges"].append(x4)
            frames[det]["right_edges"].append(x5)
            frames[det]["shifts"].append(shift)

        if plot:
            col = "C{}".format(i)
            ax.fill([
                x0, x1, frames[det_last]["left_edges"][-1],
                frames[det_last]["right_edges"][-1]
            ], [
                y0, y1, info["detector_positions"][det_last],
                info["detector_positions"][det_last]
            ],
                    alpha=0.3,
                    color=col)
            ax.plot([x0, frames[det_last]["right_edges"][-1]],
                    [y0, info["detector_positions"][det_last]],
                    color=col,
                    lw=1)
            ax.plot([x1, frames[det_last]["left_edges"][-1]],
                    [y1, info["detector_positions"][det_last]],
                    color=col,
                    lw=1)
            ax.text(0.5 * (frames[det_last]["left_edges"][-1] +
                           frames[det_last]["right_edges"][-1]),
                    info["detector_positions"][det_last],
                    "Frame {}".format(i + 1),
                    ha="center",
                    va="top")

    if plot:
        for det, dist in info["detector_positions"].items():
            # Plot detector location
            ax.plot([0, np.amax(frames[det]["right_edges"])], [dist, dist],
                    lw=3,
                    color='grey')
            ax.text(0.0, dist, det, va="bottom", ha="left")
        # Plot WFM choppers mid-point
        ax.plot([0, np.amax(frames[det_last]["right_edges"])],
                [info["wfm_choppers_midpoint"], info["wfm_choppers_midpoint"]],
                lw=1,
                color='grey',
                ls="dashed")
        ax.text(np.amax(frames[det_last]["right_edges"]),
                info["wfm_choppers_midpoint"],
                "WFM chopper mid-point",
                va="bottom",
                ha="right")
        # Save the figure
        ax.set_xlabel("Time [microseconds]")
        ax.set_ylabel("Distance [m]")
        if isinstance(plot, str):
            figname = plot
        else:
            figname = "frames_analytical.pdf"
        fig.savefig(figname, bbox_inches="tight")

    for det in frames:
        for key in frames[det]:
            frames[det][key] = np.array(frames[det][key])
        frames[det]["gaps"] = np.array([
            0.5 *
            (frames[det]["right_edges"][i] + frames[det]["left_edges"][i + 1])
            for i in range(len(frames[det]["right_edges"]) - 1)
        ])

    return frames
