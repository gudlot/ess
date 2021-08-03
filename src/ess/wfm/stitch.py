# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
import matplotlib.pyplot as plt


def _stitch_item(item, dim, frames, merge_frames, nbins):

    # if plot:
    #     fig, ax = plt.subplots()

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
                    getattr(out, group)[key] = getattr(item, group)[key]
        out.meta['source_position'] += frames["wfm_chopper_mid_point"].data
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
        # if plot:
        #     section.name = "frame-{}".format(i)
        #     # TODO: we manually remove the 2d coord here, but this should be
        #     # made into a generic check
        #     if "position" in section.coords:
        #         del section.coords["position"]
        #     elif "position" in section.attrs:
        #         del section.attrs["position"]
        #     for dim_ in list(set(section.dims) - {'tof'}):
        #         section = sc.sum(section, dim_)
        #     section.plot(ax=ax, color="C{}".format(i))

    # if plot:
    #     ax.autoscale(True)
    #     ax.relim()
    #     ax.autoscale_view()
    # if merge_frames:
    #     out.meta['source_position'] += frames["wfm_chopper_mid_point"].data

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
    dims_to_reduce = list(set(frames.dims) - {'frame'})
    for _dim in dims_to_reduce:
        frames["left_edges"] = sc.mean(frames["left_edges"], _dim)
        frames["right_edges"] = sc.mean(frames["right_edges"], _dim)

    # tof_coord = sc.linspace(
    #     dim="tof",
    #     start=(frames["left_edges"]["frame", 0] -
    #            frames["shifts"]["frame", 0]).value,
    #     stop=(frames["right_edges"]["frame", -1] -
    #           frames["shifts"]["frame", -1]).value,
    #     num=nbins + 1,
    #     unit=frames["left_edges"].unit,
    # )

    is_dataset = sc.is_dataset(data)

    # if is_dataset:
    #     key = list(data.keys())[0]
    #     dims = data[key].dims
    #     shape = data[key].shape
    # else:
    #     dims = data.dims
    #     shape = data.shape
    # dims.remove(dim)
    # shape.remove(data.sizes[dim])

    # if merge_frames:
    #     # Make empty data container
    #     empty = sc.DataArray(data=sc.zeros(dims=["tof"] + dims,
    #                                        shape=[nbins] + shape,
    #                                        variances=True,
    #                                        unit=sc.units.counts),
    #                          coords={"tof": tof_coord})
    #     for key in data.coords:
    #         if key != dim:
    #             empty.coords[key] = data.coords[key]
    #     if is_dataset:
    #         stitched = sc.Dataset()
    # else:
    #     stitched = {}

    if sc.is_dataset(data):
        if merge_frames:
            stitched = sc.Dataset()
        else:
            stitched = {}
        for i, (key, item) in enumerate(data.items()):
            # if merge_frames:
            #     for attr in item.attrs:
            #         if attr != dim:
            #             empty.attrs[attr] = item.attrs[attr]
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

    # Make sure to shift the position of the source to the midpoint between the
    # WFM choppers
    stitched.meta['source_position'] += frames["wfm_chopper_mid_point"].data

    return stitched
