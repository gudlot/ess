import scipp as sc
import scippneutron as scn
import numpy as np

def load_isis(filename, spectrum_size):
    #TODO: decide how to handle internal arguments
    tof_bins = sc.Variable(dims=['tof'], unit=sc.units.us, values=np.linspace(0, 100000, num=101))
    events = scn.load(filename=filename, mantid_args={'LoadMonitors':True})
    return sc.histogram(events['spectrum',:spectrum_size//2], tof_bins)

def load_rkh_wav(filename):
    return scn.load(
           filename=filename,
           mantid_alg='LoadRKH',
           mantid_args={'FirstColumnValue':'Wavelength'})

def load_rkh_q(filename):
    return scn.load(
           filename=filename,
           mantid_alg='LoadRKH',
           mantid_args={'FirstColumnValue':'MomentumTransfer'})

def load_mask(idf_filename, mask_file):

    return scn.load(filename=idf_filename, mantid_alg='LoadMask', mantid_args={'InputFile': mask_file})