# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
import numpy as np
from ..choppers import Chopper, ChopperKind


def make_fake_beamline(
        chopper_positions={
            "WFMC1": sc.vector(value=[0.0, 0.0, 6.775], unit='m'),
            "WFMC2": sc.vector(value=[0.0, 0.0, 7.225], unit='m')
        },
        frequency=sc.scalar(56.0, unit=sc.units.one / sc.units.s),
        lambda_min=sc.scalar(1.0, unit='angstrom'),
        pulse_length=sc.scalar(2.86e-03, unit='s'),
        pulse_t_0=sc.scalar(1.3e-4, unit='s'),
        nframes=2):
    """
    Fake chopper cascade with 2 optically blind WFM choppers.
    Based on mathematical description in Schmakat et al. (2020);
    https://www.sciencedirect.com/science/article/pii/S0168900220308640
    """

    dim = 'frame'
    # Neutron mass to Planck constant ratio
    # TODO: would be nice to use physical constants in scipp or scippneutron
    alpha = 2.5278e-4 * (sc.Unit('s') / sc.Unit('angstrom') / sc.Unit('m'))
    omega = (2.0 * np.pi * sc.units.rad) * frequency

    choppers = {}

    opening_angles_center_wfm_1 = sc.empty(dims=[dim], shape=[nframes], unit='rad')
    opening_angles_center_wfm_2 = sc.empty_like(opening_angles_center_wfm_1)
    opening_angles_width = sc.empty_like(opening_angles_center_wfm_1)

    for i in range(nframes):
        # Equation (3) in Schmakat et al. (2020)
        lambda_max = (pulse_length +
                      alpha * lambda_min * sc.norm(chopper_positions["WFMC1"])) / (
                          alpha * sc.norm(chopper_positions["WFMC2"]))
        # Equation (4) in Schmakat et al. (2020)
        theta = omega * (
            pulse_length + alpha *
            (lambda_min - lambda_max) * sc.norm(chopper_positions["WFMC1"]))
        # Equation (5) in Schmakat et al. (2020)
        phi_wfm_1 = omega * (
            pulse_t_0 + 0.5 * pulse_length + 0.5 * alpha *
            (lambda_min + lambda_max) * sc.norm(chopper_positions["WFMC1"]))
        # Equation (6) in Schmakat et al. (2020)
        phi_wfm_2 = omega * (pulse_t_0 + 1.5 * pulse_length + 0.5 * alpha * (
            (3.0 * lambda_min) - lambda_max) * sc.norm(chopper_positions["WFMC1"]))

        opening_angles_width[dim, i] = theta
        opening_angles_center_wfm_1[dim, i] = phi_wfm_1
        opening_angles_center_wfm_2[dim, i] = phi_wfm_2

        lambda_min = lambda_max

    choppers = {
        "WFMC1":
        Chopper(frequency=frequency,
                phase=sc.scalar(0.0, unit='deg'),
                position=chopper_positions["WFMC1"],
                opening_angles_center=opening_angles_center_wfm_1,
                opening_angles_width=opening_angles_width,
                kind=ChopperKind.WFM),
        "WFMC2":
        Chopper(frequency=frequency,
                phase=sc.scalar(0.0, unit='deg'),
                position=chopper_positions["WFMC2"],
                opening_angles_center=opening_angles_center_wfm_2,
                opening_angles_width=opening_angles_width,
                kind=ChopperKind.WFM),
    }

    return {
        'choppers': sc.scalar(choppers),
        'position': sc.vector(value=[0., 0., 60.], unit='m'),
        "source_pulse_length": sc.to_unit(pulse_length, 'us'),
        "source_pulse_t_0": sc.to_unit(pulse_t_0, 'us'),
        "source_position": sc.vector(value=[0.0, 0.0, 0.0], unit='m')
    }
