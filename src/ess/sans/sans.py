# SANS specific functions
import scipp as sc
import scippneutron as scn
from .contrib import map_to_bins, to_bin_centers
from .reduction import reduce_to_q, simple_reducer, grouping_reducer
from .normalization import solid_angle, transmission_fraction, covert_and_rebin


def to_wavelength(
    data,
    transmission,
    direct_beam,
    direct,
    masks,
    wavelength_bins,
    tof_bins_monitors,
    pixel_size,
    pixel_length,
):
    """
    TOF to wavelength conversion
    """
    data = data.copy(deep=False)

    (
        min_bin_incident_monitor,
        max_bin_incident_monitor,
        min_bin_transmission_monitor,
        max_bin_transmission_monitor,
    ) = tof_bins_monitors

    transmission_frac = transmission_fraction(
        sample=transmission,
        direct=direct,
        wavelength_bins=wavelength_bins,
        min_bin_incident_monitor=min_bin_incident_monitor,
        max_bin_incident_monitor=max_bin_incident_monitor,
        min_bin_transmission_monitor=min_bin_transmission_monitor,
        max_bin_transmission_monitor=max_bin_transmission_monitor,
    )

    for name, mask in masks.items():
        data.masks[name] = mask
    data = scn.convert(data, "tof", "wavelength", scatter=True)
    data = sc.rebin(data, "wavelength", wavelength_bins)
    monitor = data.attrs["monitor2"].value
    monitor = covert_and_rebin(
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
    tof_bins_monitors,
    pixel_size,
    pixel_length,
    wavelength_bins,
    wavelength_bands=None,
    groupby=None,
):
    """
    Main reduction function TOF->wavelenght->q
    """
    wav = to_wavelength(
        data=data,
        transmission=transmission,
        direct_beam=direct_beam,
        direct=direct,
        masks=masks,
        wavelength_bins=wavelength_bins,
        tof_bins_monitors=tof_bins_monitors,
        pixel_size=pixel_size,
        pixel_length=pixel_length,
    )
    reducer = simple_reducer(dim="spectrum")

    if wavelength_bands == None:
        return reduce_to_q(wav, q_bins=q_bins, reducer=reducer)
    else:
        if groupby != None:
            reducer = grouping_reducer(dim="spectrum", group=groupby)

        return reduce_to_q(
            wav, q_bins=q_bins, reducer=reducer, wavelength_bands=wavelength_bands
        )


def normalize_and_subtract(sample, background):
    """
    Normalizing and substracting background from sample
    """
    sample_norm = sample["data"] / sample["norm"]
    background_norm = background["data"] / background["norm"]
    return sample_norm - background_norm
