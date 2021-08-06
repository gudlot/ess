# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import ess.wfm as wfm
import scipp as sc
import numpy as np
import itertools
from .common import make_coords, make_default_parameters

# def _make_fake_beamline(chopper_positions, frequency, lambda_min, pulse_length,
#                         pulse_t_0, nframes):
#     """
#     Fake chopper cascade with 2 optically blind WFM choppers.
#     Based on mathematical description in Schmakat et al. (2020);
#     https://www.sciencedirect.com/science/article/pii/S0168900220308640
#     """

#     dim = 'frame'
#     # Neutron mass to Planck constant ratio
#     alpha = 2.5278e-4 * (sc.Unit('s') / sc.Unit('angstrom') / sc.Unit('m'))
#     omega = (2.0 * np.pi * sc.units.rad) * frequency

#     choppers = {}

#     opening_angles_center_1 = None
#     opening_angles_center_2 = None
#     opening_angles_width = None

#     for i in range(nframes):
#         lambda_max = (pulse_length +
#                       alpha * lambda_min * sc.norm(chopper_positions["WFMC1"])) / (
#                           alpha * sc.norm(chopper_positions["WFMC2"]))
#         theta = omega * (
#             pulse_length + alpha *
#             (lambda_min - lambda_max) * sc.norm(chopper_positions["WFMC1"]))

#         phi_wfm_1 = omega * (
#             pulse_t_0 + 0.5 * pulse_length + 0.5 * alpha *
#             (lambda_min + lambda_max) * sc.norm(chopper_positions["WFMC1"]))
#         phi_wfm_2 = omega * (pulse_t_0 + 1.5 * pulse_length + 0.5 * alpha * (
#             (3.0 * lambda_min) - lambda_max) * sc.norm(chopper_positions["WFMC1"]))

#         if opening_angles_width is None:
#             opening_angles_width = theta
#         else:
#             opening_angles_width = sc.concatenate(opening_angles_width, theta, dim)
#         if opening_angles_center_1 is None:
#             opening_angles_center_1 = phi_wfm_1
#             opening_angles_center_2 = phi_wfm_2
#         else:
#             opening_angles_center_1 = sc.concatenate(opening_angles_center_1, phi_wfm_1,
#                                                      dim)
#             opening_angles_center_2 = sc.concatenate(opening_angles_center_2, phi_wfm_2,
#                                                      dim)

#         lambda_min = lambda_max

#     choppers = {
#         "WFMC1":
#         Chopper(frequency=frequency,
#                 phase=sc.scalar(0.0, unit='deg'),
#                 position=chopper_positions["WFMC1"],
#                 opening_angles_center=opening_angles_center_1,
#                 opening_angles_width=opening_angles_width),
#         "WFMC2":
#         Chopper(frequency=frequency,
#                 phase=sc.scalar(0.0, unit='deg'),
#                 position=chopper_positions["WFMC2"],
#                 opening_angles_center=opening_angles_center_2,
#                 opening_angles_width=opening_angles_width),
#     }

#     source = {
#         "pulse_length": sc.to_unit(pulse_length, 'us'),
#         "pulse_t_0": sc.to_unit(pulse_t_0, 'us'),
#         "source_position": sc.vector(value=[0.0, 0.0, 0.0], unit='m')
#     }

#     return Beamline(choppers=choppers, source=source)

# def make_coords(**kwargs):
#     beamline = _make_fake_beamline(**kwargs)
#     chopper_cascade = wfm.make_chopper_cascade(beamline)
#     coords = {
#         'choppers': sc.scalar(chopper_cascade),
#         'position': sc.vector(value=[0., 0., 60.], unit='m')
#     }
#     for key, value in beamline.source.items():
#         coords[key] = value
#     return coords

# def make_default_parameters():
#     return {
#         "chopper_positions": {
#             "WFMC1": sc.vector(value=[0.0, 0.0, 6.775], unit='m'),
#             "WFMC2": sc.vector(value=[0.0, 0.0, 7.225], unit='m')
#         },
#         "frequency": sc.scalar(56.0, unit=sc.units.one / sc.units.s),
#         "lambda_min": sc.scalar(1.0, unit='angstrom'),
#         "pulse_length": sc.to_unit(sc.scalar(2.86e+03, unit='us'), 's'),
#         "pulse_t_0": sc.to_unit(sc.scalar(130.0, unit='us'), 's'),
#         "nframes": 2
#     }


def _frames_from_slopes(data):
    pos_norm = sc.norm(data.meta["position"])
    source_pos = sc.norm(data.meta["source_position"])

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
    # Mid-point between WFM choppers
    z_wfm = 0.5 * sc.norm(near_wfm_chopper_position + far_wfm_chopper_position)
    # Ratio of WFM chopper distances
    z_ratio_wfm = (sc.norm(far_wfm_chopper_position) /
                   sc.norm(near_wfm_chopper_position))

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
        assert sc.allclose(reference[key].data, frames[key].data)
    for i in range(frames.sizes['frame'] - 1):
        assert sc.allclose(frames["right_dt"]["frame", i].data,
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
