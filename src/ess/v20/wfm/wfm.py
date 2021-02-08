from .frames_analytical import frames_analytical
from .frames_peakfinding import frames_peakfinding


def get_frames(instrument=None, plot=False, **kwargs):
    """
    For a supplied instrument chopper cascade and detector positions, find
    the locations in microseconds of the WFM frames.

    TODO: Currently, only the analytical (time-distance) method has been tested
    and is enabled.
    The peak-finding method is temporarily disabled.
    """

    # if data is not None:
    #     return frames_peakfinding(data=data,
    #                               instrument=instrument,
    #                               plot=plot,
    #                               **kwargs)
    # else:

    return frames_analytical(instrument=instrument, plot=plot, **kwargs)
