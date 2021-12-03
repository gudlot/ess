# SANS specific functions
import scipp as sc
import scippneutron as scn
from scippneutron.tof.conversions import beamline, elastic
from .contrib import map_to_bins, to_bin_centers
from .reduction import reduce_to_q, simple_reducer, grouping_reducer
from .normalization import solid_angle, transmission_fraction, convert_and_rebin

def to_wavelength(ds):
    """
    Setting up transform coordinates graphs and convert data set for monitor to wavelength
    """
    graph = {**beamline(scatter=True), **elastic("tof")}
    ds_wav = ds.transform_coords("wavelength", graph=graph)
    graph_monitor = {**beamline(scatter=False), **elastic("tof")}
    for key in ds.keys():
        for m in ['monitor2', 'monitor4']:
            ds_wav[key].attrs[m].value = ds[key].attrs[m].value.transform_coords("wavelength", graph=graph_monitor)
    return ds_wav

def reduce_data_set(
    data,
    transmission,
    direct_beam,
    direct,
    masks,
    wavelength_bins,
    monitors_ranges,
    pixel_size,
    pixel_length,
):
    """
    TOF to wavelength conversion
    """
    #data = data.copy(deep=False)

    (
        min_bin_incident_monitor,
        max_bin_incident_monitor,
        min_bin_transmission_monitor,
        max_bin_transmission_monitor,
    ) = monitors_ranges

    transmission_frac = transmission_fraction(
        sample=transmission,
        direct=direct,
        wavelength_bins=wavelength_bins,
        min_bin_incident_monitor=min_bin_incident_monitor,
        max_bin_incident_monitor=max_bin_incident_monitor,
        min_bin_transmission_monitor=min_bin_transmission_monitor,
        max_bin_transmission_monitor=max_bin_transmission_monitor,
    )
    #TODO: Are masks necessary here?
    for name, mask in masks.items():
        data.masks[name] = mask

    #TODO: Rebin doesn't work here
    data = sc.rebin(data, "wavelength", wavelength_bins)

    monitor = data.attrs["monitor2"].value
    monitor = convert_and_rebin(
        monitor, wavelength_bins, min_bin_incident_monitor, max_bin_incident_monitor
    )

    direct_beam = map_to_bins(direct_beam, "wavelength", monitor.coords["wavelength"])
    direct_beam = monitor * transmission_frac * direct_beam

    d = sc.Dataset(
        {
            "data": data,
            "norm": solid_angle(data, pixel_size, pixel_length) * direct_beam,
        }
    )
    to_bin_centers(d, "wavelength")
    return d


def to_q(
    data,
    transmission,
    direct_beam,
    direct,
    masks,
    q_bins,
    monitors_ranges,
    pixel_size,
    pixel_length,
    wavelength_bins,
    wavelength_bands=None,
    groupby=None,
):
    """
    Main reduction function TOF->wavelenght->q
    """
    wav = reduce_data_set(
        data=data,
        transmission=transmission,
        direct_beam=direct_beam,
        direct=direct,
        masks=masks,
        wavelength_bins=wavelength_bins,
        monitors_ranges=monitors_ranges,
        pixel_size=pixel_size,
        pixel_length=pixel_length,
    )

    if groupby is None:
      reducer = simple_reducer(dim="spectrum")
    else:
      reducer = grouping_reducer(dim="spectrum", group=groupby)
    return reduce_to_q(wav, q_bins=q_bins, reducer=reducer, wavelength_bands=wavelength_bands)


def normalize_and_subtract(sample, background):
    """
    Normalizing and substracting background from sample
    """
    sample_norm = sample["data"] / sample["norm"]
    background_norm = background["data"] / background["norm"]
    return sample_norm - background_norm
