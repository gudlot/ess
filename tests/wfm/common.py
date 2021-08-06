# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
import numpy as np
import ess.wfm as wfm
from ess.wfm.choppers import Chopper
from ess.wfm.beamline import Beamline


def _to_angular_frequency(f):
    return (2.0 * np.pi) * f


# TODO replace with sc.allclose after 0.8 scipp release
def allclose(x, y):
    return sc.all(sc.isclose(x, y)).value


def _chopper_ang_freq(window_opening_t, window_size):
    ratio_of_window = window_size / (np.pi * 2)
    # Required operational frequency of chopper
    chopper_frequency = _to_angular_frequency(ratio_of_window / window_opening_t)
    return chopper_frequency


def _make_fake_beamline(chopper_positions, frequency, lambda_min, pulse_length,
                        pulse_t_0, nframes):
    """
    Fake chopper cascade with 2 optically blind WFM choppers.
    Based on mathematical description in Schmakat et al. (2020);
    https://www.sciencedirect.com/science/article/pii/S0168900220308640
    """

    dim = 'frame'
    # Neutron mass to Planck constant ratio
    alpha = 2.5278e-4 * (sc.Unit('s') / sc.Unit('angstrom') / sc.Unit('m'))
    omega = (2.0 * np.pi * sc.units.rad) * frequency

    choppers = {}

    opening_angles_center_1 = None
    opening_angles_center_2 = None
    opening_angles_width = None

    for i in range(nframes):
        lambda_max = (pulse_length +
                      alpha * lambda_min * sc.norm(chopper_positions["WFMC1"])) / (
                          alpha * sc.norm(chopper_positions["WFMC2"]))
        theta = omega * (
            pulse_length + alpha *
            (lambda_min - lambda_max) * sc.norm(chopper_positions["WFMC1"]))

        phi_wfm_1 = omega * (
            pulse_t_0 + 0.5 * pulse_length + 0.5 * alpha *
            (lambda_min + lambda_max) * sc.norm(chopper_positions["WFMC1"]))
        phi_wfm_2 = omega * (pulse_t_0 + 1.5 * pulse_length + 0.5 * alpha * (
            (3.0 * lambda_min) - lambda_max) * sc.norm(chopper_positions["WFMC1"]))

        if opening_angles_width is None:
            opening_angles_width = theta
        else:
            opening_angles_width = sc.concatenate(opening_angles_width, theta, dim)
        if opening_angles_center_1 is None:
            opening_angles_center_1 = phi_wfm_1
            opening_angles_center_2 = phi_wfm_2
        else:
            opening_angles_center_1 = sc.concatenate(opening_angles_center_1, phi_wfm_1,
                                                     dim)
            opening_angles_center_2 = sc.concatenate(opening_angles_center_2, phi_wfm_2,
                                                     dim)

        lambda_min = lambda_max

    choppers = {
        "WFMC1":
        Chopper(frequency=frequency,
                phase=sc.scalar(0.0, unit='deg'),
                position=chopper_positions["WFMC1"],
                opening_angles_center=opening_angles_center_1,
                opening_angles_width=opening_angles_width),
        "WFMC2":
        Chopper(frequency=frequency,
                phase=sc.scalar(0.0, unit='deg'),
                position=chopper_positions["WFMC2"],
                opening_angles_center=opening_angles_center_2,
                opening_angles_width=opening_angles_width),
    }

    source = {
        "pulse_length": sc.to_unit(pulse_length, 'us'),
        "pulse_t_0": sc.to_unit(pulse_t_0, 'us'),
        "source_position": sc.vector(value=[0.0, 0.0, 0.0], unit='m')
    }

    return Beamline(choppers=choppers, source=source)


def make_coords(**kwargs):
    beamline = _make_fake_beamline(**kwargs)
    chopper_cascade = wfm.make_chopper_cascade(beamline)
    coords = {
        'choppers': sc.scalar(chopper_cascade),
        'position': sc.vector(value=[0., 0., 60.], unit='m')
    }
    for key, value in beamline.source.items():
        coords[key] = value
    return coords


def make_default_parameters():
    return {
        "chopper_positions": {
            "WFMC1": sc.vector(value=[0.0, 0.0, 6.775], unit='m'),
            "WFMC2": sc.vector(value=[0.0, 0.0, 7.225], unit='m')
        },
        "frequency": sc.scalar(56.0, unit=sc.units.one / sc.units.s),
        "lambda_min": sc.scalar(1.0, unit='angstrom'),
        "pulse_length": sc.to_unit(sc.scalar(2.86e+03, unit='us'), 's'),
        "pulse_t_0": sc.to_unit(sc.scalar(130.0, unit='us'), 's'),
        "nframes": 2
    }
