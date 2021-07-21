import scipp as sc
import numpy as np


def _to_angular_frequency(f):
    return (2.0 * np.pi) * f


# TODO replace with sc.allclose after 0.8 scipp release
def allclose(x, y):
    return sc.all(sc.isclose(x, y)).value


def _chopper_ang_freq(window_opening_t, window_size):
    ratio_of_window = window_size / (np.pi * 2)
    # Required operational frequency of chopper
    chopper_frequency = _to_angular_frequency(ratio_of_window /
                                              window_opening_t)
    return chopper_frequency
