import scipp as sc
import scippneutron as scn


def solid_angle(data, pixel_size, pixel_length):
    """
    Solid angle function taking pixel_size and pixel_lenght as parameters
    """
    L2 = scn.L2(data)
    return (pixel_size * pixel_length) / (L2 * L2)


def transmission_fraction(
    sample,
    direct,
    wavelength_bins,
    min_bin_mon2,
    max_bin_mon2,
    min_bin_mon4,
    max_bin_mon4,
):
    """
    Approximation based on equations in CalculateTransmission documentation
    p = \frac{S_T}{D_T}\frac{D_I}{S_I}
    This is equivalent to mantid.CalculateTransmission without fitting

    Note: TOF ranges fro
    """

    def setup(data, begin, end, scatter):
        transformed = data - sc.mean(data["tof", begin:end], "tof")
        transformed = scn.convert(transformed, "tof", "wavelength", scatter=scatter)
        transformed = sc.rebin(transformed.copy(), "wavelength", wavelength_bins)
        return transformed

    us = sc.units.us

    sample_incident = setup(
        sample.attrs["monitor2"].value, min_bin_mon2, max_bin_mon2, scatter=False
    )
    sample_trans = setup(
        sample.attrs["monitor4"].value, min_bin_mon4, max_bin_mon4, scatter=False
    )
    direct_incident = setup(
        direct.attrs["monitor2"].value, min_bin_mon2, max_bin_mon2, scatter=False
    )
    direct_trans = setup(
        direct.attrs["monitor4"].value, min_bin_mon4, max_bin_mon4, scatter=False
    )

    return (sample_trans / direct_trans) * (direct_incident / sample_incident)
