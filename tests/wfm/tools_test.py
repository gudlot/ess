# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
import numpy as np
from ess.wfm.tools import _angular_frame_edge_to_time


def test_angular_frame_edge_to_time():
    omega = sc.scalar(np.pi, unit='rad/s')
    angle = sc.scalar(0.5 * np.pi, unit='rad')
    phase = sc.scalar(0., unit='rad')

    assert sc.allclose(_angular_frame_edge_to_time(omega, angle, phase),
                       sc.to_unit(sc.scalar(0.5, unit='s'), 'us'))

    phase = sc.scalar(2.0 * np.pi / 3.0, unit='rad')
    assert sc.allclose(_angular_frame_edge_to_time(omega, angle, phase),
                       sc.to_unit(sc.scalar(0.5 + 2.0 / 3.0, unit='s'), 'us'))
