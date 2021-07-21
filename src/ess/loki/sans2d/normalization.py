import scipp as sc
import scippneutron as scn

def solid_angle(data):
    # TODO proper solid angle
    # [0.0117188,0.0075,0.0075] bounding box size
    pixel_size = 0.055 * sc.units.m
    #1.05m / 512 pixels
    pixel_length = 0.004 * sc.units.m
    L2 = scn.L2(data)
    return (pixel_size * pixel_length) / (L2 * L2)


def transmission_fraction(sample, direct, wavelength_bins):
    # Approximation based on equations in CalculateTransmission documentation
    # p = \frac{S_T}{D_T}\frac{D_I}{S_I}
    # This is equivalent to mantid.CalculateTransmission without fitting
    def setup(data, begin, end, scatter):
        background = data - sc.mean(data['tof', begin:end], 'tof')
        background = scn.convert(background, 'tof', 'wavelength', scatter=scatter)
        #TODO: check rebining
        background = sc.rebin(background.copy(), 'wavelength', wavelength_bins)
        return background

    us = sc.units.us
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
