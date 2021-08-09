# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import ess.wfm as wfm
import scipp as sc
from .common import make_coords, make_default_parameters, allclose


def _frames_from_slopes(data):
    pos_norm = sc.norm(data.meta["position"])

    # Get the number of WFM frames
    nframes = data.meta["choppers"].value["opening_angles_open"].sizes["frame"]

    # Now find frame boundaries
    frames = sc.Dataset()
    frames["left_edges"] = sc.zeros(dims=["frame"], shape=[nframes], unit=sc.units.us)
    frames["right_edges"] = sc.zeros_like(frames["left_edges"])
    frames["left_dt"] = sc.zeros_like(frames["left_edges"])
    frames["right_dt"] = sc.zeros_like(frames["left_edges"])
    frames["shifts"] = sc.zeros(dims=["frame"], shape=[nframes], unit=sc.units.us)

    near_wfm_chopper_position = data.meta["choppers"].value["position"]["chopper",
                                                                        0].data
    far_wfm_chopper_position = data.meta["choppers"].value["position"]["chopper",
                                                                       1].data

    # Distance between WFM choppers
    dz_wfm = sc.norm(far_wfm_chopper_position - near_wfm_chopper_position)

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

        frames["left_edges"]["frame", i] = t_lambda_min
        frames["left_dt"]["frame", i] = dt_lambda_min
        frames["right_edges"]["frame", i] = t_lambda_max
        frames["right_dt"]["frame", i] = dt_lambda_max
        frames["shifts"]["frame", i] = tstart["chopper", 1]

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
        assert allclose(frames["right_dt"]["frame", i].data,
                        frames["left_dt"]["frame", i + 1].data)


def test_frames_analytical():
    ds = sc.Dataset(coords=make_coords(**make_default_parameters()))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_large_dz_wfm():
    params = make_default_parameters()
    params["chopper_positions"] = {
        "WFMC1": sc.vector(value=[0.0, 0.0, 6.0], unit='m'),
        "WFMC2": sc.vector(value=[0.0, 0.0, 8.0], unit='m')
    }
    ds = sc.Dataset(coords=make_coords(**params))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_short_pulse():
    params = make_default_parameters()
    params["pulse_length"] = sc.to_unit(sc.scalar(1.86e+03, unit='us'), 's')
    ds = sc.Dataset(coords=make_coords(**params))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_large_t_0():
    params = make_default_parameters()
    params["pulse_t_0"] = sc.to_unit(sc.scalar(300., unit='us'), 's')
    ds = sc.Dataset(coords=make_coords(**params))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_6_frames():
    params = make_default_parameters()
    params["nframes"] = 6
    ds = sc.Dataset(coords=make_coords(**params))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_short_lambda_min():
    params = make_default_parameters()
    params["lambda_min"] = sc.scalar(0.5, unit='angstrom')
    ds = sc.Dataset(coords=make_coords(**params))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)
