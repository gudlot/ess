# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippneutron as scn


def load_sans2d(filename: str, spectrum_size: int,
                tof_bins: sc.Variable) -> sc.DataArray:
    """
    Loading wrapper for ISIS SANS2D files
    """
    events = scn.load(filename=filename, mantid_args={"LoadMonitors": True})
    return sc.bin(events["spectrum", :spectrum_size], edges=[tof_bins])


def load_rkh_wav(filename: str) -> sc.DataArray:
    """
    Loading wrapper for RKH files
    """
    return scn.load(filename=filename,
                    mantid_alg="LoadRKH",
                    mantid_args={"FirstColumnValue": "Wavelength"})


def load_rkh_q(filename: str) -> sc.DataArray:
    """
    Loading wrapper for RKH files
    """
    return scn.load(filename=filename,
                    mantid_alg="LoadRKH",
                    mantid_args={"FirstColumnValue": "MomentumTransfer"})
