# standard library imports
import itertools
import os
from typing import Optional

# related third party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from IPython.display import display, HTML
# possibilites for median filters, I did not benchmark, apparently median_filter
# could be the faster and medfilt2d is faster than medfilt
from scipy.ndimage import median_filter
from scipy.signal import medfilt
from scipy.optimize import leastsq  # needed for fitting of turbidity

# local application imports
import scippneutron as scn
import scippnexus as snx
import scipp as sc


def split_sample_dark_reference(da):
    """Separate incoming dataarray into the three contributions: sample, dark, reference.

    Parameters
    ----------
    da: scipp.DataArray
            sc.DataArray that contains spectroscopy contributions sample, dark, reference

    Returns:
    ----------
    da_dict: dict
            Dictionary that contains spectroscopy data signal (data) from the sample,
            the reference, and the dark measurement.
            Keys: sample, reference, dark

    """
    assert isinstance(da, sc.DataArray)

    dark = da[da.coords["is_dark"]].squeeze()
    ref = da[da.coords["is_reference"]].squeeze()
    sample = da[da.coords["is_data"]]    
       
    #TODO Instead of a dict a sc.Dataset? 
    return {"sample": sample, "reference": ref, "dark": dark}