# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ..wfm.choppers import Chopper, ChopperKind


def make_beamline() -> dict:
    """
    ODIN chopper cascade and component positions.
    Chopper opening angles taken from Schmakat et al. (2020)
    https://www.sciencedirect.com/science/article/pii/S0168900220308640

    Note that the values listed in the paper for the FOC1 opening angles are wrong.
    The correct values are used here.
    """

    dim = 'frame'
    hz = sc.units.one / sc.units.s

    choppers = {
        "WFMC1":
        Chopper(frequency=sc.scalar(56.0, unit=hz),
                phase=sc.scalar(0.0, unit='deg'),
                position=sc.vector(value=[0, 0, 6.775], unit='m'),
                opening_angles_center=sc.array(
                    dims=[dim],
                    values=[91.93, 142.23, 189.40, 233.63, 275.10, 313.99],
                    unit='deg'),
                opening_angles_width=sc.array(
                    dims=[dim],
                    values=[5.74, 8.98, 12.01, 14.85, 17.52, 20.02],
                    unit='deg'),
                kind=ChopperKind.WFM),
        "WFMC2":
        Chopper(frequency=sc.scalar(56.0, unit=hz),
                phase=sc.scalar(0.0, unit='deg'),
                position=sc.vector(value=[0, 0, 7.225], unit='m'),
                opening_angles_center=sc.array(
                    dims=[dim],
                    values=[97.67, 151.21, 201.41, 248.48, 292.62, 334.01],
                    unit='deg'),
                opening_angles_width=sc.array(
                    dims=[dim],
                    values=[5.74, 8.98, 12.01, 14.85, 17.52, 20.02],
                    unit='deg'),
                kind=ChopperKind.WFM),
        "FOC1":
        Chopper(frequency=sc.scalar(42.0, unit=hz),
                phase=sc.scalar(0.0, unit='deg'),
                position=sc.vector(value=[0, 0, 8.4], unit='m'),
                opening_angles_center=sc.array(
                    dims=[dim],
                    values=[81.12, 127.82, 171.60, 212.66, 251.16, 288.85],
                    unit='deg'),
                opening_angles_width=sc.array(
                    dims=[dim],
                    values=[11.06, 13.06, 14.94, 16.70, 18.36, 19.91],
                    unit='deg'),
                kind=ChopperKind.FRAME_OVERLAP),
        "FOC2":
        Chopper(frequency=sc.scalar(42.0, unit=hz),
                phase=sc.scalar(0.0, unit='deg'),
                position=sc.vector(value=[0, 0, 12.20], unit='m'),
                opening_angles_center=sc.array(
                    dims=[dim],
                    values=[106.57, 174.42, 238.04, 297.53, 353.48, 46.65 + 360.0],
                    unit='deg'),
                opening_angles_width=sc.array(
                    dims=[dim],
                    values=[32.90, 33.54, 34.15, 34.71, 35.24, 35.74],
                    unit='deg'),
                kind=ChopperKind.FRAME_OVERLAP),
        "FOC3":
        Chopper(frequency=sc.scalar(28.0, unit=hz),
                phase=sc.scalar(0.0, unit='deg'),
                position=sc.vector(value=[0, 0, 17.0], unit='m'),
                opening_angles_center=sc.array(
                    dims=[dim],
                    values=[92.47, 155.52, 214.65, 270.09, 322.08, 11.39 + 360.0],
                    unit='deg'),
                opening_angles_width=sc.array(
                    dims=[dim],
                    values=[40.32, 39.61, 38.94, 38.31, 37.72, 37.16],
                    unit='deg'),
                kind=ChopperKind.FRAME_OVERLAP),
        "FOC4":
        Chopper(frequency=sc.scalar(14.0, unit=hz),
                phase=sc.scalar(0.0, unit='deg'),
                position=sc.vector(value=[0, 0, 23.69], unit='m'),
                opening_angles_center=sc.array(
                    dims=[dim],
                    values=[61.17, 105.11, 146.32, 184.96, 221.19, 255.72],
                    unit='deg'),
                opening_angles_width=sc.array(
                    dims=[dim],
                    values=[32.98, 31.82, 30.74, 29.72, 28.77, 27.87],
                    unit='deg'),
                kind=ChopperKind.FRAME_OVERLAP),
        "FOC5":
        Chopper(frequency=sc.scalar(14.0, unit=hz),
                phase=sc.scalar(0.0, unit='deg'),
                position=sc.vector(value=[0, 0, 33.0], unit='m'),
                opening_angles_center=sc.array(
                    dims=[dim],
                    values=[82.20, 143.05, 200.44, 254.19, 304.68, 353.46],
                    unit='deg'),
                opening_angles_width=sc.array(
                    dims=[dim],
                    values=[50.81, 48.55, 46.42, 44.43, 42.56, 40.80],
                    unit='deg'),
                kind=ChopperKind.FRAME_OVERLAP)
    }

    source = {
        "source_pulse_length": sc.scalar(2.86e+03, unit='us'),
        "source_pulse_t_0": sc.scalar(130.0, unit='us'),
        "source_position": sc.vector(value=[0.0, 0.0, 0.0], unit='m')
    }

    return {"choppers": choppers, "source": source}
