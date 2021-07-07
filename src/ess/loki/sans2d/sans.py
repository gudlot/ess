# SANS specific functions
import numpy as np
import scipp as sc
import scippneutron
import contrib
import reduction


def project_xy(data, nx=100, ny=20):
    z = sc.geometry.z(scippneutron.position(data))
    x = sc.geometry.x(scippneutron.position(data)) / z
    y = sc.geometry.y(scippneutron.position(data)) / z
    data.coords['x/z'] = x
    data.coords['y/z'] = y
    x = sc.Variable(dims=['x/z'],
                    values=np.linspace(sc.min(x).value,
                                       sc.max(x).value,
                                       num=nx))
    y = sc.Variable(dims=['y/z'],
                    values=np.linspace(sc.min(y).value,
                                       sc.max(y).value,
                                       num=ny))
    return sc.realign(data, coords={'y/z': y, 'x/z': x})


def solid_angle(data):
    # TODO proper solid angle
    # [0.0117188,0.0075,0.0075] bounding box size
    pixel_size = 0.005 * sc.units.m
    #1.05m / 512 pixels
    pixel_length = 0.002 * sc.units.m
    L2 = scippneutron.L2(data)
    return (pixel_size * pixel_length) / (L2 * L2)


def subtract_background_mean(data, dim, begin, end):
    coord = data.coords[dim]
    assert (coord.unit == begin.unit) and (coord.unit == end.unit)
    i = np.searchsorted(coord.values, begin.value)
    j = np.searchsorted(coord.values, end.value) + 1
    return data - sc.mean(data[dim, i:j], dim)


def transmission_fraction(sample, direct, wavelength_bins):
    # Approximation based on equations in CalculateTransmission documentation
    # p = \frac{S_T}{D_T}\frac{D_I}{S_I}
    # This is equivalent to mantid.CalculateTransmission without fitting
    def setup(data, begin, end, scatter):
        background = subtract_background_mean(data, 'tof', begin, end)
        #del background.coords['sample_position']  # ensure unit conversion treats this a monitor
        background = scippneutron.convert(background, 'tof', 'wavelength', scatter=scatter)
        background = sc.rebin(background, 'wavelength', wavelength_bins)
        return background

    us = sc.units.us
    #sample_incident = setup(sample['spectrum', 0], 40000.0 * us, 99000.0 * us)
    #sample_trans = setup(sample['spectrum', 3], 88000.0 * us, 98000.0 * us)
    #direct_incident = setup(direct['spectrum', 0], 40000.0 * us, 99000.0 * us)
    #direct_trans = setup(direct['spectrum', 3], 88000.0 * us, 98000.0 * us)

    #TODO: resolve this
    sample_incident = setup(sample.attrs['monitor2'].value, 85000.0 * us, 98000.0 * us, scatter=False)
    sample_trans = setup(sample.attrs['monitor4'].value, 50.0 * us, 3000.0 * us, scatter=False)
    direct_incident = setup(direct.attrs['monitor2'].value, 85000.0 * us, 98000.0 * us, scatter=False)
    direct_trans = setup(direct.attrs['monitor4'].value, 50.0 * us, 3000.0 * us, scatter=False)

    return (sample_trans / direct_trans) * (direct_incident / sample_incident)
    #CalculateTransmission(SampleRunWorkspace=transWsTmp,
    #                      DirectRunWorkspace=transWsTmp,
    #                      OutputWorkspace=outWsName,
    #                      IncidentBeamMonitor=1,
    #                      TransmissionMonitor=4, RebinParams='0.9,-0.025,13.5',
    #                      FitMethod='Polynomial',
    #                      PolynomialOrder=3, OutputUnfittedData=True)


def to_wavelength(data, transmission, direct_beam, direct_beam_transmission,
                  masks, wavelength_bins):
    data = data.copy()
    transmission = transmission_fraction(transmission,
                                         direct_beam_transmission,
                                         wavelength_bins)
    for name, mask in masks.items():
        data.masks[name] = mask
    data = scippneutron.convert(data, 'tof', 'wavelength', out=data, scatter=True)
    data = sc.rebin(data, 'wavelength', wavelength_bins)

    monitor = data.attrs['monitor2'].value
    monitor = subtract_background_mean(monitor, 'tof', 85000.0 * sc.units.us,
                                       98000.0 * sc.units.us)

    monitor = scippneutron.convert(monitor, 'tof', 'wavelength', out=monitor, scatter=False)
    monitor = sc.rebin(monitor, 'wavelength', wavelength_bins)

    direct_beam = contrib.map_to_bins(direct_beam, 'wavelength',
                                      monitor.coords['wavelength'])
    direct_beam = monitor * transmission * direct_beam

    d = sc.Dataset({'data': data, 'norm': solid_angle(data) * direct_beam})
    contrib.to_bin_centers(d, 'wavelength')
    return d


def to_q(data, transmission, direct_beam, direct_beam_transmission, masks, q_bins,
         wavelength_bins, wavelength_bands=None, groupby=None):
    wav = to_wavelength(data=data,
                             transmission=transmission,
                             direct_beam=direct_beam,
                             direct_beam_transmission=direct_beam_transmission,
                             masks=masks,
                             wavelength_bins=wavelength_bins)
    reducer = reduction.simple_reducer(dim='spectrum')

    if wavelength_bands == None:
        return reduction.reduce_to_q(wav, q_bins=q_bins,
                                     reducer=reducer)
    else:
        # TODO: Check if this the case only when one does slices or in general
        if groupby != None:
            reducer = reduction.grouping_reducer(dim='spectrum', group=groupby)

        return reduction.reduce_to_q(wav, q_bins=q_bins,
                                     reducer=reducer,
                                     wavelength_bands=wavelength_bands)

def normalize_and_subtract(sample, background):
    sample_norm = sample['data'] / sample['norm']
    background_norm = background['data'] / background['norm']
    return sample_norm - background_norm