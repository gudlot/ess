import scipp as sc
import scippneutron as scn


def solid_angle(data, pixel_size, pixel_length):
    """
    Solid angle function taking pixel_size and pixel_lenght as parameters.
    The method assumes pixels are rectangular.
    """
    L2 = scn.L2(data)
    return (pixel_size * pixel_length) / (L2 * L2)


#TODO: Change name of function once all issues are sorted out
def substract_background_and_rebin(data, wavelength_bins, threshold):
    """
    Shifts data by backgorund value and rebins it
    """
    data_original = data.copy(deep=False)
    data_original.masks['background'] = data_original.data > threshold
    background = sc.mean(data_original)
    transformed = data - background
    transformed = sc.rebin(transformed, "wavelength", wavelength_bins)
    return transformed

def transmission_fraction(
    sample,
    direct,
    wavelength_bins,
    threshold
):
    """
    Approximation based on equations in CalculateTransmission documentation
    p = \frac{S_T}{D_T}\frac{D_I}{S_I}
    This is equivalent to mantid.CalculateTransmission without fitting
    """

    sample_incident = substract_background_and_rebin(
        sample.attrs["monitor2"].value,
        wavelength_bins,
        threshold
    )
    sample_trans = substract_background_and_rebin(
        sample.attrs["monitor4"].value,
        wavelength_bins,
        threshold
    )
    direct_incident = substract_background_and_rebin(
        direct.attrs["monitor2"].value,
        wavelength_bins,
        threshold
    )
    direct_trans = substract_background_and_rebin(
        direct.attrs["monitor4"].value,
        wavelength_bins,
        threshold
    )
    return (sample_trans / direct_trans) * (direct_incident / sample_incident)
