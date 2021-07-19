# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
from ..wfm.choppers import Beamline, Chopper


def make_beamline():

    dim = 'frame'
    hz = sc.units.one / sc.units.s

    choppers = {
        "WFMC1":
        Chopper(frequency=sc.scalar(56.0, unit=hz),
                phase=sc.scalar(0.0, unit='deg'),
                distance=sc.vector(value=[0, 0, 6.775], unit='m'),
                frame_center=sc.array(
                    dims=[dim],
                    values=[91.93, 142.23, 189.40, 233.63, 275.10, 313.99],
                    unit='deg'),
                frame_width=sc.array(
                    dims=[dim],
                    values=[5.70, 9.00, 12.00, 14.90, 17.50, 20.00],
                    unit='deg')),
        "WFMC2":
        Chopper(frequency=sc.scalar(56.0, unit=hz),
                phase=sc.scalar(0.0, unit='deg'),
                distance=sc.vector(value=[0, 0, 7.225], unit='m'),
                frame_center=sc.array(
                    dims=[dim],
                    values=[97.67, 151.21, 201.41, 248.48, 292.62, 334.01],
                    unit='deg'),
                frame_width=sc.array(
                    dims=[dim],
                    values=[5.70, 9.00, 12.00, 14.90, 17.50, 20.00],
                    unit='deg')),
        "FOC1":
        Chopper(frequency=sc.scalar(42.0, unit=hz),
                phase=sc.scalar(0.0, unit='deg'),
                distance=sc.vector(value=[0, 0, 8.4], unit='m'),
                frame_center=sc.array(
                    dims=[dim],
                    values=[81.12, 127.82, 171.60, 212.66, 251.16, 288.85],
                    unit='deg'),
                frame_width=sc.array(
                    dims=[dim],
                    values=[32.90, 33.54, 34.15, 34.37, 34.89, 34.31],
                    unit='deg')),
        "FOC2":
        Chopper(frequency=sc.scalar(42.0, unit=hz),
                phase=sc.scalar(0.0, unit='deg'),
                distance=sc.vector(value=[0, 0, 12.20], unit='m'),
                frame_center=sc.array(dims=[dim],
                                      values=[
                                          106.57, 174.42, 238.04, 297.53,
                                          353.48, 46.65 + 360.0
                                      ],
                                      unit='deg'),
                frame_width=sc.array(
                    dims=[dim],
                    values=[32.90, 33.54, 34.15, 34.37, 34.89, 34.31],
                    unit='deg')),
        "FOC3":
        Chopper(frequency=sc.scalar(28.0, unit=hz),
                phase=sc.scalar(0.0, unit='deg'),
                distance=sc.vector(value=[0, 0, 17.0], unit='m'),
                frame_center=sc.array(dims=[dim],
                                      values=[
                                          92.47, 155.52, 214.65, 270.09,
                                          322.08, 11.39 + 360.0
                                      ],
                                      unit='deg'),
                frame_width=sc.array(
                    dims=[dim],
                    values=[40.32, 39.61, 38.94, 38.31, 37.72, 36.05],
                    unit='deg')),
        "FOC4":
        Chopper(frequency=sc.scalar(14.0, unit=hz),
                phase=sc.scalar(0.0, unit='deg'),
                distance=sc.vector(value=[0, 0, 23.69], unit='m'),
                frame_center=sc.array(
                    dims=[dim],
                    values=[61.17, 105.11, 146.32, 184.96, 221.19, 255.72],
                    unit='deg'),
                frame_width=sc.array(
                    dims=[dim],
                    values=[32.98, 31.82, 30.74, 29.72, 28.77, 26.76],
                    unit='deg')),
        "FOC5":
        Chopper(frequency=sc.scalar(14.0, unit=hz),
                phase=sc.scalar(0.0, unit='deg'),
                distance=sc.vector(value=[0, 0, 33.0], unit='m'),
                frame_center=sc.array(
                    dims=[dim],
                    values=[81.94, 143.17, 200.11, 254.19, 304.47, 353.76],
                    unit='deg'),
                frame_width=sc.array(
                    dims=[dim],
                    values=[50.81, 48.55, 45.49, 41.32, 37.45, 37.74],
                    unit='deg'))
    }

    source = {
        "pulse_length": sc.scalar(2.86e+03, unit='us'),
        "pulse_t_0": sc.scalar(140.0, unit='us'),
        "source_position": sc.vector(value=[0.0, 0.0, 0.0], unit='m')
    }

    return Beamline(choppers=choppers, source=source)
