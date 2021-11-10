import scipp as sc
import scippneutron as scn


def solid_angle(data, pixel_size, pixel_length):
    """
    Solid angle function taking pixel_size and pixel_lenght as parameters.
    The method assumes pixels are rectangular.
    """
    L2 = scn.L2(data)
    return (pixel_size * pixel_length) / (L2 * L2)


def covert_and_rebin(data, wavelength_bins, begin, end):
    """
    Convertining and rebinning shifted data
    """
    transformed = data - sc.mean(data["tof", begin:end], "tof")
    transformed = scn.convert(transformed, "tof", "wavelength", scatter=False)
    transformed = sc.rebin(transformed, "wavelength", wavelength_bins)
    return transformed


def transmission_fraction(
    sample,
    direct,
    wavelength_bins,
    min_bin_incident_monitor,
    max_bin_incident_monitor,
    min_bin_transmission_monitor,
    max_bin_transmission_monitor,
):
    """
    Approximation based on equations in CalculateTransmission documentation
    p = \frac{S_T}{D_T}\frac{D_I}{S_I}
    This is equivalent to mantid.CalculateTransmission without fitting
    """

    sample_incident = covert_and_rebin(
        sample.attrs["monitor2"].value,
        wavelength_bins,
        min_bin_incident_monitor,
        max_bin_incident_monitor,
    )
    sample_trans = covert_and_rebin(
        sample.attrs["monitor4"].value,
        wavelength_bins,
        min_bin_transmission_monitor,
        max_bin_transmission_monitor,
    )
    direct_incident = covert_and_rebin(
        direct.attrs["monitor2"].value,
        wavelength_bins,
        min_bin_incident_monitor,
        max_bin_incident_monitor,
    )
    direct_trans = covert_and_rebin(
        direct.attrs["monitor4"].value,
        wavelength_bins,
        min_bin_transmission_monitor,
        max_bin_transmission_monitor,
    )

    return (sample_trans / direct_trans) * (direct_incident / sample_incident)
