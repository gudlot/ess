# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import ess.wfm as wfm
import scipp as sc
from .common import allclose


def _frames_from_slopes(data):
    pos_norm = sc.norm(data.meta["position"])

    # Get the number of WFM frames
    nframes = data.meta["choppers"].value["opening_angles_open"].sizes["frame"]

    # Now find frame boundaries
    frames = sc.Dataset()
    frames["time_min"] = sc.zeros(dims=["frame"], shape=[nframes], unit=sc.units.us)
    frames["time_max"] = sc.zeros_like(frames["time_min"])
    frames["delta_time_min"] = sc.zeros_like(frames["time_min"])
    frames["delta_time_max"] = sc.zeros_like(frames["time_min"])
    frames["wavelength_min"] = sc.zeros(dims=["frame"],
                                        shape=[nframes],
                                        unit=sc.units.angstrom)
    frames["wavelength_max"] = sc.zeros_like(frames["wavelength_min"])
    frames["delta_wavelength_min"] = sc.zeros_like(frames["wavelength_min"])
    frames["delta_wavelength_max"] = sc.zeros_like(frames["wavelength_min"])

    frames["time_correction"] = sc.zeros(dims=["frame"],
                                         shape=[nframes],
                                         unit=sc.units.us)

    near_wfm_chopper_position = data.meta["choppers"].value["position"]["chopper",
                                                                        0].data
    far_wfm_chopper_position = data.meta["choppers"].value["position"]["chopper",
                                                                       1].data

    # Distance between WFM choppers
    dz_wfm = sc.norm(far_wfm_chopper_position - near_wfm_chopper_position)
    # Mid-point between WFM choppers
    z_wfm = 0.5 * sc.norm(near_wfm_chopper_position + far_wfm_chopper_position)
    # Neutron mass to Planck constant ratio
    # TODO: would be nice to use physical constants in scipp or scippneutron
    alpha = 2.5278e+2 * (sc.Unit('us') / sc.Unit('angstrom') / sc.Unit('m'))

    for i in range(nframes):
        tstart, tend = wfm.frame_opening_and_closing_times(
            data.meta["choppers"].value["frame", i])
        dt_lambda_max = tend['chopper', 0] - tstart['chopper', 0]
        slope_lambda_max = dz_wfm / dt_lambda_max
        intercept_lambda_max = sc.norm(
            near_wfm_chopper_position) - slope_lambda_max * tend['chopper', 0]
        t_lambda_max = (pos_norm - intercept_lambda_max) / slope_lambda_max

        slope_lambda_min = sc.norm(near_wfm_chopper_position) / (
            tend['chopper', 0] - (data.meta["pulse_length"] + data.meta["pulse_t_0"]))
        intercept_lambda_min = sc.norm(
            far_wfm_chopper_position) - slope_lambda_min * tstart['chopper', 1]
        t_lambda_min = (pos_norm - intercept_lambda_min) / slope_lambda_min

        t_lambda_min_plus_dt = (pos_norm -
                                (sc.norm(near_wfm_chopper_position) - slope_lambda_min *
                                 tend['chopper', 0])) / slope_lambda_min
        dt_lambda_min = t_lambda_min_plus_dt - t_lambda_min

        # Compute wavelength information
        lambda_min = (t_lambda_min + 0.5 * dt_lambda_min -
                      tstart["chopper", 1]) / (alpha * (pos_norm - z_wfm))
        lambda_max = (t_lambda_max - 0.5 * dt_lambda_max -
                      tstart["chopper", 1]) / (alpha * (pos_norm - z_wfm))
        dlambda_min = dz_wfm * lambda_min / (pos_norm - z_wfm)
        dlambda_max = dz_wfm * lambda_max / (pos_norm - z_wfm)

        frames["time_min"]["frame", i] = t_lambda_min
        frames["delta_time_min"]["frame", i] = dt_lambda_min
        frames["time_max"]["frame", i] = t_lambda_max
        frames["delta_time_max"]["frame", i] = dt_lambda_max
        frames["wavelength_min"]["frame", i] = lambda_min
        frames["wavelength_max"]["frame", i] = lambda_max
        frames["delta_wavelength_min"]["frame", i] = dlambda_min
        frames["delta_wavelength_max"]["frame", i] = dlambda_max
        frames["time_correction"]["frame", i] = tstart["chopper", 1]

    frames["wfm_chopper_mid_point"] = sc.mean(
        sc.concatenate(data.meta["choppers"].value["position"]["chopper", 0],
                       data.meta["choppers"].value["position"]["chopper", 1], 'none'))
    return frames


def _check_against_reference(ds, frames):
    reference = _frames_from_slopes(ds)
    for key in frames:
        # TODO: once scipp 0.8 is released, use sc.allclose here which also works on
        # vector_3_float64.
        # assert allclose(reference[key].data, frames[key].data)
        if frames[key].dtype == sc.dtype.vector_3_float64:
            for xyz in "xyz":
                assert allclose(getattr(reference[key].data.fields, xyz),
                                getattr(frames[key].data.fields, xyz))
        else:
            assert allclose(reference[key].data, frames[key].data)
    for i in range(frames.sizes['frame'] - 1):
        assert allclose(frames["delta_time_max"]["frame", i].data,
                        frames["delta_time_min"]["frame", i + 1].data)


def test_frames_analytical():
    ds = sc.Dataset(coords=wfm.make_fake_beamline())
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_large_dz_wfm():
    ds = sc.Dataset(coords=wfm.make_fake_beamline(
        chopper_positions={
            "WFMC1": sc.vector(value=[0.0, 0.0, 6.0], unit='m'),
            "WFMC2": sc.vector(value=[0.0, 0.0, 8.0], unit='m')
        }))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_short_pulse():
    ds = sc.Dataset(coords=wfm.make_fake_beamline(
        pulse_length=sc.to_unit(sc.scalar(1.86e+03, unit='us'), 's')))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_large_t_0():
    ds = sc.Dataset(coords=wfm.make_fake_beamline(
        pulse_t_0=sc.to_unit(sc.scalar(300., unit='us'), 's')))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_6_frames():
    ds = sc.Dataset(coords=wfm.make_fake_beamline(nframes=6))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_short_lambda_min():
    ds = sc.Dataset(coords=wfm.make_fake_beamline(
        lambda_min=sc.scalar(0.5, unit='angstrom')))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)
