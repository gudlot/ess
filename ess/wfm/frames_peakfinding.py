"""
  Find WFM frames using peak-finding.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def _tof_shifts(pscdata, psc_frequency=0):
    cut_out_centre = np.reshape(pscdata * 180.0 / np.pi,
                                (len(pscdata) // 2, 2)).mean(1)
    cut_out_diffs = np.ediff1d(cut_out_centre)
    return cut_out_diffs / (360.0 * psc_frequency)


def _get_frame_shifts(instrument):
    """
    This function generates a list of proper shifting parameters from psc data.

    :param instrument: An instrument setup, containing chopper information.
    :return: List of relative frame shifts in microseconds.
    """

    # Factor of 0.5 * 1.0e6 comes from taking mean and converting to
    # microseconds
    relative_shifts = (_tof_shifts(
        instrument["choppers"]["WFM1"].openings,
        psc_frequency=instrument["choppers"]["WFM1"].frequency) + _tof_shifts(
            instrument["choppers"]["WFM2"].openings,
            psc_frequency=instrument["choppers"]["WFM2"].frequency)) * 5.0e+05
    return -relative_shifts


def _make_frame_shifts(initial_shift, other_shifts=None):
    """
    This function constructs a list of proper shifting parameters for
    wavelength frames.
    It expects the initial shift in microseconds, followed by an optional
    list/tuple of
    shifts relative to the previous one. For example:

        frame_shifts = make_frame_shifts(-6000, [-2000, -2000])

    results in -6000,-8000,-10000.

    The default other_shifts are the currently correct shifts for the V20
    beamline at HZB.

    :param initial_shift: Shift to apply to first wavelength frame in
                          microseconds.
    :param other_shifts: Iterable of shift increments with respect to the
                         previous value (initial_shift for first value in
                         list), also in microseconds.
    :return: List of absolute frame shifts in microseconds.
    """
    frame_shift_increments = [initial_shift] + list(other_shifts)
    frame_shifts = [
        sum(frame_shift_increments[:i + 1])
        for i in range(len(frame_shift_increments))
    ]

    print("The frame_shifts are:", frame_shifts)

    return frame_shifts


def frames_peakfinding(data=None,
                       instrument=None,
                       initial_shift=-6630,
                       plot=False,
                       verbose=False,
                       nbins=512,
                       exclude_left=None,
                       exclude_right=None,
                       bg_threshold=0.05,
                       peak_prominence=0.05,
                       gsmooth=2,
                       inter_frame_threshold=1.5,
                       inter_frame_gaps=None,
                       window_threshold=0.3):
    """


    """

    frame_shifts = _make_frame_shifts(initial_shift,
                                      _get_frame_shifts(instrument))

    if "events" in data:
        # Find min and max in event time-of-arrival (TOA)
        xmin = np.amin(data["events"])
        xmax = np.amax(data["events"])
        nevents = len(data["events"])
        print("Read in {} events.".format(nevents))
        print("The minimum and maximum tof values are:", xmin, xmax)
        # Add padding
        dx = (xmax - xmin) / float(nbins)
        xmin -= dx
        xmax += dx
        frames_x = np.linspace(xmin, xmax, nbins + 1)
        y, edges = np.histogram(data["events"], bins=frames_x)
        x = 0.5 * (edges[:-1] + edges[1:])
    else:
        y = data["y"]
        x = data["x"]

    if plot:
        # Histogram the events for plotting
        fig, ax = plt.subplots()
        ax.plot(x, y, color='k', label="Raw data", lw=3)

    # Gaussian smooth the data
    y = gaussian_filter1d(y, gsmooth)
    if plot:
        ax.plot(x, y, color="lightgray", label="Gaussian smoothed", lw=1)

    # Minimum and maximum values in the histogram
    ymin = np.amin(y)
    ymax = np.amax(y)

    # Determine background threshold by histogramming data
    nx = len(x)
    hist, edges = np.histogram(y, bins=50)
    bg_raw = edges[np.argmax(hist) + 1]
    bg_value = bg_raw + bg_threshold * (ymax - bg_raw)
    # Find the leading and trailing edges; i.e. the leftmost and rightmost
    # points that exceed the background value
    i_start = exclude_left if exclude_left is not None else 0
    i_end = exclude_right if exclude_right is not None else nx - 1
    for i in range(i_start, nx):
        if y[i] > bg_value:
            i_start = i
            break
    for i in range(i_end, 1, -1):
        if y[i] > bg_value:
            i_end = i
            break

    # Find valleys with scipy.signal.find_peaks by inverting the y array
    peaks, params = find_peaks(-y,
                               prominence=peak_prominence * (ymax - ymin),
                               distance=nbins // 20)

    # Check for unwanted peaks:
    # 1. If a peak is found inside the noise leading the signal, it will often
    # have a zero left base
    if exclude_left is None:
        exclude_left = nbins // 10
    to_be_removed = []
    for p in range(len(peaks)):
        if params["left_bases"][p] < exclude_left:
            print("Removed peak number {} at x,y = {},{} because of a bad "
                  "left base".format(p, x[peaks[p]], y[peaks[p]]))
            to_be_removed.append(p)
    # 2. If there are peaks in the middle of a frame, then the y value is high
    threshold = inter_frame_threshold * bg_value
    for p in range(len(peaks)):
        if y[peaks[p]] > threshold:
            print("Removed peak number {} at x,y = {},{} because the y value "
                  "exceeds the threshold {}".format(
                    p, x[peaks[p]], y[peaks[p]], threshold))
            to_be_removed.append(p)
    # Actually remove the peaks
    peaks = np.delete(peaks, to_be_removed)

    if len(peaks) != instrument["info"]["nframes"] - 1:
        raise RuntimeError(
            "The number of valleys found does not match the "
            "required number of frames. Expected "
            "{}, got {}.".format(instrument["info"]["nframes"] - 1,
                                 len(peaks)))

    frame_gaps = x[peaks]
    print("The frame_gaps are:", frame_gaps)

    peaks = np.concatenate([[i_start], peaks, [i_end]])

    # Compute boundaries
    frame_boundaries = np.zeros([len(frame_gaps) + 1, 2])
    frame_boundaries[0, 0] = x[i_start]
    frame_boundaries[-1, 1] = x[i_end]
    if inter_frame_gaps is not None:
        # Use constant gap width between frames
        if not isinstance(inter_frame_gaps, list):
            inter_frame_gaps = [inter_frame_gaps] * len(frame_gaps)
        for i, g in enumerate(frame_gaps):
            frame_boundaries[i, 1] = g - 0.5 * inter_frame_gaps[i]
            frame_boundaries[i + 1, 0] = g + 0.5 * inter_frame_gaps[i]
    else:
        # Use window threshold method
        window_means = np.zeros(instrument["info"]["nframes"])
        for i in range(instrument["info"]["nframes"]):
            window_means[i] = np.mean(y[peaks[i]:peaks[i + 1]] -
                                      bg_raw) * window_threshold
        for j in range(1, len(peaks) - 1):
            # Iterate backwards
            for i in range(peaks[j], peaks[j - 1], -1):
                if (y[i] - bg_raw) >= window_means[j - 1]:
                    frame_boundaries[j - 1, 1] = x[i]
                    break
            # Iterate forwards
            for i in range(peaks[j], peaks[j + 1]):
                if (y[i] - bg_raw) >= window_means[j]:
                    frame_boundaries[j, 0] = x[i]
                    break
    print("The frame boundaries are:", frame_boundaries)

    # Plot the diagnostics
    if plot:
        ax.plot(x[peaks], y[peaks], 'o', color='r', zorder=10)
        ax.axhline(threshold, ls="dashed", color="g")
        ax.axhline(bg_raw, ls="solid", color="pink", lw=1)
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        for i in range(instrument["info"]["nframes"]):
            col = "C{}".format(i)
            ax.axvspan(frame_boundaries[i, 0],
                       frame_boundaries[i, 1],
                       color=col,
                       alpha=0.4)
        ax.axhspan(bg_value, yl[0], color='gray', alpha=0.4)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        if isinstance(plot, str):
            figname = plot
        else:
            figname = "frames_peakfinding.pdf"
        fig.savefig(figname, bbox_inches="tight")

    frames = {
        "left_edges": np.array([f[0] for f in frame_boundaries]),
        "right_edges": np.array([f[1] for f in frame_boundaries]),
        "gaps": np.array(frame_gaps),
        "shifts": np.array(frame_shifts)
    }

    return frames
