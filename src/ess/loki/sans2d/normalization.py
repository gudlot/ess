import scipp as sc
import scippneutron as scn


def solid_angle(data, pixel_size, pixel_length):
    """
    Solid angle function taking pixel_size and pixel_lenght as parameters
    """
    L2 = scn.L2(data)
    return (pixel_size * pixel_length) / (L2 * L2)


def transmission_fraction(
    sample, background, direct, wavelength_bins, min_bin, max_bin
):
    """
    Approximation based on equations in CalculateTransmission documentation
    p = \frac{S_T}{D_T}\frac{D_I}{S_I}
    This is equivalent to mantid.CalculateTransmission without fitting
    """

    def setup(data, begin, end, scatter):
        transformed = data - sc.mean(data["tof", begin:end], "tof")
        transformed = scn.convert(transformed, "tof", "wavelength", scatter=scatter)
        transformed = sc.rebin(transformed.copy(), "wavelength", wavelength_bins)
        return transformed

    us = sc.units.us
    # TODO: resolve TOF ranges and make sure these are actually correct sample atributes used
    sample_incident = setup(
        sample.attrs["monitor2"].value, min_bin, max_bin, scatter=False
    )
    sample_trans = setup(
        sample.attrs["monitor4"].value, 50.0 * us, 3000.0 * us, scatter=False
    )
    direct_incident = setup(
        direct.attrs["monitor2"].value, min_bin, max_bin, scatter=False
    )
    direct_trans = setup(
        direct.attrs["monitor4"].value, 50.0 * us, 3000.0 * us, scatter=False
    )
    # direct_incident = setup(background.attrs['monitor2'].value, 85000.0 * us, 98000.0 * us, scatter=False)
    # irect_trans = setup(background.attrs['monitor4'].value, 50.0 * us, 3000.0 * us, scatter=False)

    return (sample_trans / direct_trans) * (direct_incident / sample_incident)
