# standard library imports
import itertools
import os
from typing import Optional, Type

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


def load_nurfloki_file(name: str, exp_meth: str ):
    """ Loads data of a specified experimental method from the corresponding entry in a
     NUrF-Loki.nxs file. 
    

    Parameters
    ----------
    name: str
        Filename, e.g. 066017.nxs
    exp_meth: str
        Experimental method available with the NUrF exp. configuration.
        Current default values: uv, fluorescence. TODO in the future: raman

    Returns
    ----------
    exp_meth_dict: dict
        Dictionary of sc.DataArrays. Keys: data, reference, dark. 
        Data contains all relevant signals of the sample.

    """
    nurf_meth=['uv', 'fluorescence']

    if not isinstance(exp_meth, str):
        raise TypeError('exp_math needs to be of type str.')

    if not exp_meth in nurf_meth:
        raise ValueError('Wrong string. This method does not exist for NurF at LoKi.')

    path_to_group=f"entry/instrument/{exp_meth}"
    
    with snx.File(name) as fnl:
        meth = fnl[path_to_group][()]

    # separation
    exp_meth_dict = split_sample_dark_reference(meth)

    return exp_meth_dict