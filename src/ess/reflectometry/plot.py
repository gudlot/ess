# flake8: noqa: E501
"""
Some very simple plotting code for use in reflectometry reduction.
"""
import scipp as sc
from ess.reflectometry import binning


def log_R(data, labels=None, ymin=1e-6, ymax=1, q_bin_kwargs=None):
    """
    Plot intensity against q.

    Args:
        data (`list` of `ReflData`): A list of ReflData objects for plotting.
        labels (`list` of `str`): Labels for each of the ReflData objects.
        ymin (`float`): Minimum value of y to plot.
        ymax (`float`): Maximum value of y to plot.
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
        plots[labels[i]] = datum.q_bin(**q_bin_kwargs)
    fig = sc.plot(plots, norm="log")
    fig.ax.set_ylim((ymin, ymax))
    return fig


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
