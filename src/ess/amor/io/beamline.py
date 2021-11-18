# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ...wfm.choppers import Chopper


def make_beamline() -> dict:
    """
    Amor beamline components.
    """
    hz = sc.units.one / sc.units.s

    beamline = {
        "source_chopper":
        sc.scalar(
            Chopper(frequency=sc.scalar(20 / 3, unit=hz),
                    phase=sc.scalar(-8.0, unit='deg'),
                    position=sc.vector(value=[0, 0, -15.0], unit='m')))
    }

    # The source position is not the true source position due to the
    # use of choppers to define the pulse.
    beamline["source_position"] = beamline["source_chopper"].value.position
    return beamline
