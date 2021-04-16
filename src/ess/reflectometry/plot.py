"""
Some very simple plotting code for use in reflectometry reduction.
"""
import scipp as sc


def plot(data, labels=None, q_bin_kwargs=None):
    """
    Plot intensity against q.

    Args:
        data (`list` of `ReflData`): A list of ReflData objects for plotting.
        labels (`list` of `str`): Labels for each of the ReflData objects.
        q_bin_kwargs (`dict`, optional): A dictionary of keyword arguments to be passed to the :py:func:`q_bin` class method. Optional, default is that default :py:func:`q_bin` keywords arguments are used.

    Returns:
        (`scipp.plotting.plot1d.SciPlot1d`): Intensity vs q plot.
    """
    try:
        _ = len(data)
    except TypeError:
        data = [data]
    if q_bin_kwargs is None:
        q_bin_kwargs = {}
    if labels is None:
        labels = [f"{i}" for i in range(len(data))]
    plots = {}
    for i, datum in enumerate(data):
        plots[labels[i]] = datum.q_bin(**q_bin_kwargs).bins.sum()
    return sc.plot(plots, norm="log")


def wavelength_theta(data):
    """
    Plot 2d histogram of wavelength against theta.

    Args:
        data (`ReflData`): ReflData object for plotting.

    Returns:
        (`scipp.plotting.plot2d.SciPlot2d`): 2d histogram of wavelength against theta.
    """
    return sc.plot(data.wavelength_theta_bin())


def wavelength_q(data):
    """
    Plot 2d histogram of wavelength against q.

    Args:
        data (`ReflData`): ReflData object for plotting.

    Returns:
        (`scipp.plotting.plot2d.SciPlot2d`): 2d histogram of wavelength against q.
    """
    return sc.plot(data.wavelength_q_bin())


def q_theta(data):
    """
    Plot 2d histogram of q against theta.

    Args:
        data (`ReflData`): ReflData object for plotting.

    Returns:
        (`scipp.plotting.plot2d.SciPlot2d`): 2d histogram of q against theta.
    """
    return sc.plot(data.q_theta_bin())
