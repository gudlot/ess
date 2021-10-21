# SANS specific functions
import scipp as sc
import scippneutron as scn
from ess.sans import contrib
from ess.sans import reduction
from ess.sans import normalization


def to_wavelength(
    data,
    background,
    transmission,
    direct_beam,
    direct_beam_transmission,
    masks,
    wavelength_bins,
    min_bin,
    max_bin,
    pixel_size,
    pixel_length,
):
    """
    TOF to wavelength conversion
    """
    data = data.copy()
    transmission = normalization.transmission_fraction(
        transmission,
        background,
        direct_beam_transmission,
        wavelength_bins,
        min_bin,
        max_bin,
    )
    for name, mask in masks.items():
        data.masks[name] = mask
    data = scn.convert(data, "tof", "wavelength", out=data, scatter=True)
    data = sc.rebin(data, "wavelength", wavelength_bins)

    monitor = data.attrs["monitor2"].value
    monitor = monitor - sc.mean(monitor["tof", min_bin:max_bin], "tof")
    monitor = scn.convert(monitor, "tof", "wavelength", out=monitor, scatter=False)
    monitor = sc.rebin(monitor, "wavelength", wavelength_bins)

    direct_beam = contrib.map_to_bins(
        direct_beam, "wavelength", monitor.coords["wavelength"]
    )
    direct_beam = monitor * transmission * direct_beam

    d = sc.Dataset(
        {
            "data": data,
            "norm": normalization.solid_angle(data, pixel_size, pixel_length)
            * direct_beam,
        }
    )
    contrib.to_bin_centers(d, "wavelength")
    return d


def to_q(
    data,
    background,
    transmission,
    direct_beam,
    direct_beam_transmission,
    masks,
    q_bins,
    min_bin,
    max_bin,
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
        background=background,
        transmission=transmission,
        direct_beam=direct_beam,
        direct_beam_transmission=direct_beam_transmission,
        masks=masks,
        wavelength_bins=wavelength_bins,
        min_bin=min_bin,
        max_bin=max_bin,
        pixel_size=pixel_size,
        pixel_length=pixel_length,
    )
    reducer = reduction.simple_reducer(dim="spectrum")

    if wavelength_bands == None:
        return reduction.reduce_to_q(wav, q_bins=q_bins, reducer=reducer)
    else:
        # TODO: Check if this the case only when one does slices or in general
        if groupby != None:
            reducer = reduction.grouping_reducer(dim="spectrum", group=groupby)

        return reduction.reduce_to_q(
            wav, q_bins=q_bins, reducer=reducer, wavelength_bands=wavelength_bands
        )


def normalize_and_subtract(sample, background):
    """
    Normalizing and substracting background from sample
    """
    sample_norm = sample["data"] / sample["norm"]
    background_norm = background["data"] / background["norm"]
    return sample_norm - background_norm
