# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
import matplotlib.pyplot as plt


def _stitch_item(item, dim, frames, merge_frames, nbins):

    if merge_frames:
        # Make empty data container
        dims = []
        shape = []
        for dim_ in item.dims:
            if dim_ != dim:
                dims.append(dim_)
                shape.append(item.sizes[dim_])
            else:
                dims.append('tof')
                shape.append(nbins)
        out = sc.DataArray(data=sc.zeros(dims=dims,
                                         shape=shape,
                                         variances=item.variances is not None,
                                         unit=item.unit),
                           coords={
                               "tof":
                               sc.linspace(
                                   dim="tof",
                                   start=(frames["left_edges"]["frame", 0] -
                                          frames["shifts"]["frame", 0]).value,
                                   stop=(frames["right_edges"]["frame", -1] -
                                         frames["shifts"]["frame", -1]).value,
                                   num=nbins + 1,
                                   unit=frames["left_edges"].unit,
                               )
                           })
        for group in ["coords", "attrs"]:
            for key in getattr(item, group):
                if key != dim:
                    getattr(out, group)[key] = getattr(item, group)[key].copy()
    else:
        out = {}

    for i in range(frames.sizes["frame"]):
        section = item[dim, frames["left_edges"].data[
            "frame", i]:frames["right_edges"].data["frame", i]].copy()
        section.meta[dim] -= frames["shifts"].data["frame", i]
        section.rename_dims({dim: 'tof'})

        if merge_frames:
            out += sc.rebin(section, 'tof', out.meta["tof"])
        else:
            section.meta['source_position'] += frames[
                "wfm_chopper_mid_point"].data
            out[f"frame{i}"] = section

    # Note: we need to do the modification here because if not there is a coordinate
    # mismatch between `out` and `section`
    if merge_frames:
        out.meta['source_position'] += frames["wfm_chopper_mid_point"].data

    return out


def stitch(data, dim, frames, merge_frames=True, nbins=256):
    """
    Convert raw arrival time WFM data to time-of-flight by shifting each frame
    (described by the `frames` argument) by a time offset defined by the position
    of the WFM choppers.
    This process is also known as 'stitching' the frames.
    """

    # TODO: for now, if frames depend on positions, we take the mean along the
    # position dimensions. We should implement the position-dependent stitching
    # in the future.
    frames = frames.copy()
    dims_to_reduce = list(set(frames.dims) - {'frame'})
    for _dim in dims_to_reduce:
        frames["left_edges"] = sc.mean(frames["left_edges"], _dim)
        frames["right_edges"] = sc.mean(frames["right_edges"], _dim)

    if sc.is_dataset(data):
        if merge_frames:
            stitched = sc.Dataset()
        else:
            stitched = {}
        for i, (key, item) in enumerate(data.items()):
            stitched[key] = _stitch_item(item=item,
                                         dim=dim,
                                         frames=frames,
                                         merge_frames=merge_frames,
                                         nbins=nbins)
    else:
        stitched = _stitch_item(item=data,
                                dim=dim,
                                frames=frames,
                                merge_frames=merge_frames,
                                nbins=nbins)

    return stitched
