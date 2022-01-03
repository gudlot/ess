# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import scipp as sc
from typing import Union
from .frames_analytical import frames_analytical
# from .frames_peakfinding import frames_peakfinding


def get_frames(data: Union[sc.DataArray, sc.Dataset], **kwargs) -> sc.Dataset:
    """
    For a supplied instrument chopper cascade and detector positions, find
    the locations in microseconds of the WFM frames.

    TODO: Currently, only the analytical (time-distance) method has been tested
    and is enabled.
    The peak-finding method is temporarily disabled.
    """

    # if data is not None:
    #     return frames_peakfinding(data=data,
    #                               instrument=instrument,
    #                               plot=plot,
    #                               **kwargs)
    # else:

    return frames_analytical(data=data, **kwargs)
