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
            sc.DataArray that contains spectroscopy contributions sample, dark,
            reference

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


def apply_medfilter(
    da: sc.DataArray, kernel_size: Optional[int] = None
) -> sc.DataArray:
    #TODO: Rewrite this function with median filters offered by scipp.
    """Applies a median filter to a sc.DataArray that contains fluo or uv spectra to
    remove spikes
    Filter used: from scipy.ndimage import median_filter. This filter function is maybe
    subject to change for another scipy function.
    Caution:  The scipy mean (median?) filter will just ignore errorbars on counts
    according to Neil Vaytet.

    Parameters
    ----------
    da: sc.DataArray
        DataArray that contains already fluo or uv spectra in data.

    kernel_size: int
        Scalar giving the size of the median filter window. Elements of kernel_size should be odd. Default size is 3

    Returns
    ----------
    da_mfilt: sc.DataArray
        New sc.DataArray with median filtered data

    """
    if kernel_size is None:
        kernel_size = 3
    else:
        assert type(kernel_size) is int, "kernel_size must be an integer"

    # check if kernel_size is odd
    if (kernel_size % 2) != 1:
        raise ValueError("kernel_size should be odd.")

    # extract all spectreal values
    data_spectrum = da.values

    # apply a median filter
    # yes, we can apply a median filter to an array (vectorize it), but the kernel needs to be adapted to the  ndim off the array
    # no filterin in the first dimension (spectrum), but in the second (wavelength)
    # magic suggested by Simon
    # this will make [kernel_size] for 1d data, and [1, kernel_size] for 2d data
    ksize = np.ones_like(da.shape)
    ksize[-1] = kernel_size
    data_spectrum_filt = median_filter(data_spectrum, size=ksize)

    # create a new sc.DataArray where the data is replaced by the median filtered data
    # make a deep copy
    da_mfilt = da.copy()

    # replace original data with filtered data
    da_mfilt.values = data_spectrum_filt

    # graphical comparison
    legend_props = {"show": True, "loc": (1.04, 0)}
    figs, axs = plt.subplots(1, 2, figsize=(16, 5))
    # out1=sc.plot(da['spectrum',7], ax=axs[0])
    # out2=sc.plot(da_mfilt['spectrum',7], ax=axs[1])
    out1 = sc.plot(
        sc.collapse(da, keep="wavelength"),
        linestyle="dashed",
        marker=".",
        grid=True,
        legend=legend_props,
        title=f"Before median filter",
        ax=axs[0],
    )
    out2 = sc.plot(
        sc.collapse(da_mfilt, keep="wavelength"),
        linestyle="dashed",
        marker=".",
        grid=True,
        legend=legend_props,
        title=f"After median filter - kernel_size: {kernel_size}",
        ax=axs[1],
    )
    # display(figs)

    return da_mfilt



def export_uv(name, path_output):
    """Export normalized all uv data and an averaged uv spectrum in a LoKI.nxs file to .dat file
    
    Attention: Current output format follows custom format for an individual user, not
    any other software.

    Parameters
    ----------
    name: str
        Filename of LoKI.nxs file that contains the UV data for export

    path_output: str
        Absolute path to output folder

    Returns
    ----------
    Tab-limited .dat file with columns wavelength, dark, reference, (multiple) raw uv spectra, (multiple) corrected uv spectra, one averaged uv spectrum

    """

    uv_dict = load_nurfloki_file(name, 'uv')  
    normalized = normalize_uv(
        **uv_dict
    ) 
   
    normalized_avg = normalized.mean("spectrum")

    # prepare for export as .dat files
    output_filename = f"{name}_uv.dat"

    # puzzle the header together
    l = "".join(
        ["dark_{0}\t".format(i) for i in range(uv_dict['dark'].ndim)
        ]
        )
    m="".join(
        ["reference_{0}\t".format(i) for i in range(uv_dict['reference'].ndim)
        ]
        )

    n="".join(
        [   
            "uv_raw_spectra_{0}\t".format(i)
            for i, x in enumerate(range(uv_dict['sample'].sizes["spectrum"]))
        ]
        )
    o="".join(    
        [   
            "uv_norm_spectra_{0}\t".format(i)
            for i, x in enumerate(range(normalized.sizes["spectrum"]))
        ]
        )
    
    p = "uv_spectra_avg\t" 
  

    hdrtxt = "wavelength [nm]\t"
    final_header = hdrtxt + l + m + n + o + p


    data_to_save = np.column_stack(
        (
            normalized.coords["wavelength"].values.transpose(),

            # raw data
            uv_dict['dark'].values.transpose(),
            uv_dict['reference'].values.transpose(),
            uv_dict['sample'].values.transpose(),

            # reduced data
            normalized.values.transpose(),
            normalized_avg.values.transpose(),
            
        )
    )
    path_to_save = os.path.join(path_output, output_filename)

    # dump the content
    with open(path_to_save, "w") as f:
        np.savetxt(f, data_to_save, fmt="%.10f", delimiter="\t", header=final_header)



def export_fluo(name, path_output):
    """Export corrected fluo data contained in a LoKI.nxs file to .dat file

    Attention: Current output format follows custom format for an individual user, not
    any other software.

    Parameters
    ----------
    name: str
        Filename of LoKI.nxs file that contains the fluo data for export

    path_output: str
        Absolute path to output folder

    Returns
    ----------
    Tab-limited .dat file with columns wavelength, dark, reference, multiple raw fluo 
    spectra, and multiple normalized fluo spectra.
    Header of each fluo spectrum column contains the incident excitation energy.

    """
    # export of all calculated fluo data in a LOKI.nxs name to .dat
    # input: filename of LOKI.nxs: name, str, path_output: absolut path to output folder, str

    fluo_dict = load_nurfloki_file(name, 'fluorescence')  
    final_fluo = normalize_fluo(**fluo_dict)

    # prepare for export as .dat files
    output_filename = f"{name}_fluo.dat"
    path_to_save = os.path.join(path_output, output_filename)

    
    
    l = "".join(
        ["dark_{0}\t".format(i) for i in range(fluo_dict['dark'].ndim)
        ]
        )
    m = "".join(
        ["reference_{0}\t".format(i) for i in range(fluo_dict['reference'].ndim)
        ]
        )
    
    n= "".join([f"raw_{i}nm\t" for i in final_fluo.coords["monowavelengths"].values])

    o= "".join([f"norm_{i}nm\t" for i in final_fluo.coords["monowavelengths"].values])



    hdrtxt = "wavelength [nm]\t"
    final_header = hdrtxt + l + m + n + o
 
    data_to_save = np.column_stack(
        (
            final_fluo.coords["wavelength"].values.transpose(),
            # dark
            fluo_dict['dark'].values.transpose(),
            # reference
            fluo_dict['reference'].values.transpose(),
            # sample
            fluo_dict['sample'].values.transpose(),
            # final fluo spectra
            final_fluo.data["spectrum", :].values.transpose(),
        )
    )

    # dump the content
    with open(path_to_save, "w") as f:
        np.savetxt(
            f,
            data_to_save,
            fmt="".join(["%.5f\t"] + ["%.5e\t"] * (fluo_dict['dark'].ndim + 
                        fluo_dict['reference'].ndim + 2*final_fluo.sizes["spectrum"])),
            delimiter="\t",
            header=final_header,
        )
