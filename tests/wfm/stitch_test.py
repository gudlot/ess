# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import ess.wfm as wfm
import numpy as np
import scipp as sc
import scippneutron as scn


def test_basic_stitching():
    frames = sc.Dataset()
    shift = -5.0
    frames['time_min'] = sc.array(dims=['frame'], values=[0.0], unit=sc.units.us)
    frames['time_max'] = sc.array(dims=['frame'], values=[10.0], unit=sc.units.us)
    frames['time_correction'] = sc.array(dims=['frame'],
                                         values=[shift],
                                         unit=sc.units.us)
    frames["wfm_chopper_mid_point"] = sc.vector(value=[0., 0., 2.0], unit='m')

    data = sc.DataArray(data=sc.ones(dims=['t'], shape=[100], unit=sc.units.counts),
                        coords={
                            't':
                            sc.linspace(dim='t',
                                        start=0.0,
                                        stop=10.0,
                                        num=101,
                                        unit=sc.units.us),
                            'source_position':
                            sc.vector(value=[0., 0., 0.], unit='m')
                        })

    nbins = 10
    stitched = wfm.stitch(data=data, dim='t', frames=frames, bins=nbins)
    # Note dimension change to TOF as well as shift
    assert sc.identical(
        sc.values(stitched),
        sc.DataArray(data=sc.ones(dims=['tof'], shape=[nbins], unit=sc.units.counts) *
                     nbins,
                     coords={
                         'tof':
                         sc.linspace(dim='tof',
                                     start=0.0 - shift,
                                     stop=10.0 - shift,
                                     num=nbins + 1,
                                     unit=sc.units.us),
                         'source_position':
                         sc.vector(value=[0., 0., 2.], unit='m')
                     }))


def _do_stitching_on_beamline(wavelengths):
    # Make beamline parameters for 6 frames
    coords = wfm.make_fake_beamline(nframes=6)

    # They are all created half-way through the pulse.
    # Compute their arrival time at the detector.
    alpha = 2.5278e-4 * (sc.Unit('s') / sc.Unit('angstrom') / sc.Unit('m'))
    dz = sc.norm(coords['position'] - coords['source_position'])
    arrival_times = sc.to_unit(
        alpha * dz * wavelengths,
        'us') + coords['source_pulse_t_0'] + (0.5 * coords['source_pulse_length'])

    # Make a data array that contains the beamline and the time coordinate
    tmin = sc.min(arrival_times)
    tmax = sc.max(arrival_times)
    dt = 0.1 * (tmax - tmin)
    coords['time'] = sc.linspace(dim='time',
                                 start=(tmin - dt).value,
                                 stop=(tmax + dt).value,
                                 num=2001,
                                 unit=dt.unit)
    counts, _ = np.histogram(arrival_times.values, bins=coords['time'].values)
    da = sc.DataArray(coords=coords,
                      data=sc.array(dims=['time'], values=counts, unit='counts'))

    # Find location of frames
    frames = wfm.get_frames(da)

    stitched = wfm.stitch(frames=frames, data=da, dim='time', bins=2001)

    wav = scn.convert(stitched, origin='tof', target='wavelength', scatter=False)
    rebinned = sc.rebin(wav,
                        dim='wavelength',
                        bins=sc.linspace(dim='wavelength',
                                         start=1.0,
                                         stop=10.0,
                                         num=1001,
                                         unit='angstrom'))

    choppers = da.meta["choppers"].value
    # near_wfm_chopper_position = da.meta["choppers"].value["position"]["chopper", 0].data
    # far_wfm_chopper_position = da.meta["choppers"].value["position"]["chopper", 1].data
    # Distance between WFM choppers
    dz_wfm = sc.norm(choppers["WFMC2"].position - choppers["WFMC1"].position)
    # Delta_lambda  / lambda
    dlambda_over_lambda = dz_wfm / sc.norm(coords['position'] -
                                           frames['wfm_chopper_mid_point'].data)

    return rebinned, dlambda_over_lambda


def _check_lambda_inside_resolution(lam, dlam_over_lam, data, check_value=True):
    dlam = 0.5 * dlam_over_lam * lam
    assert sc.isclose(
        sc.sum(data['wavelength', lam - dlam:lam + dlam]).data,
        1.0 * sc.units.counts).value is check_value


def test_stitching_on_beamline():
    # Create 6 neutrons with selected wavelengths, one neutron per frame
    wavelengths = sc.array(dims=['wavelength'],
                           values=[1.75, 3.2, 4.5, 6.0, 7.0, 8.25],
                           unit='angstrom')
    rebinned, dlambda_over_lambda = _do_stitching_on_beamline(wavelengths)

    for i in range(len(wavelengths)):
        _check_lambda_inside_resolution(wavelengths['wavelength', i],
                                        dlambda_over_lambda, rebinned)


def test_stitching_on_beamline_bad_wavelength():
    # Create 6 neutrons. The first wavelength is in this case too short to pass through
    # the WFM choppers.
    wavelengths = sc.array(dims=['wavelength'],
                           values=[1.5, 3.2, 4.5, 6.0, 7.0, 8.25],
                           unit='angstrom')
    rebinned, dlambda_over_lambda = _do_stitching_on_beamline(wavelengths)

    # The first wavelength should fail the check, since anything not passing through
    # the choppers won't satisfy the dlambda/lambda condition.
    _check_lambda_inside_resolution(wavelengths['wavelength', 0],
                                    dlambda_over_lambda,
                                    rebinned,
                                    check_value=False)
    for i in range(1, len(wavelengths)):
        _check_lambda_inside_resolution(wavelengths['wavelength', i],
                                        dlambda_over_lambda, rebinned)
