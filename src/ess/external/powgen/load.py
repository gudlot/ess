# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
"""
File loading for POWGEN.
"""
from pathlib import Path
from typing import Union

import scipp as sc
import scippneutron as scn


def load(filename: Union[str, Path]) -> sc.DataArray:
    """
    Load a data file for POWGEN.

    Parameters
    ----------
    filename:
        Input file name.

    Returns
    -------
    da:
        Loaded data.
    """
    return scn.load(
        filename,
        advanced_geometry=True,
        load_pulse_times=False,
        mantid_args={"LoadMonitors": True},
    )
