# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
import matplotlib.pyplot as plt


def _stitch_item(item, dim, frames, target, plot):

    if plot:
        fig, ax = plt.subplots()

    for i in range(frames.sizes["frame"]):
        section = item[dim, frames["left_edges"].data[
            "frame", i]:frames["right_edges"].data["frame", i]].copy()
        section.meta[dim] -= frames["shifts"].data["frame", i]
        section.rename_dims({dim: 'tof'})

        target += sc.rebin(section, 'tof', target.meta["tof"])
        if plot:
            section.name = "frame-{}".format(i)
            # TODO: we manually remove the 2d coord here, but this should be
            # made into a generic check
            if "position" in section.coords:
                del section.coords["position"]
            elif "position" in section.attrs:
                del section.attrs["position"]
            for dim_ in list(set(section.dims) - {'tof'}):
                section = sc.sum(section, dim_)
            section.plot(ax=ax, color="C{}".format(i))

    if plot:
        ax.autoscale(True)
        ax.relim()
        ax.autoscale_view()

    return target


def stitch(data, dim, frames, nbins=256, plot=False):

    # TODO: for now, if frames depend on positions, we take the mean along the
    # position dimensions. We should implement the position-dependent stitching
    # in the future.
    dims_to_reduce = list(set(frames.dims) - {'frame'})
    for _dim in dims_to_reduce:
        frames["left_edges"] = sc.mean(frames["left_edges"], _dim)
        frames["right_edges"] = sc.mean(frames["right_edges"], _dim)

    tof_coord = sc.linspace(
        dim="tof",
        start=(frames["left_edges"]["frame", 0] -
               frames["shifts"]["frame", 0]).value,
        stop=(frames["right_edges"]["frame", -1] -
              frames["shifts"]["frame", -1]).value,
        num=nbins + 1,
        unit=frames["left_edges"].unit,
    )

    is_dataset = sc.is_dataset(data)

    if is_dataset:
        key = list(data.keys())[0]
        dims = data[key].dims
        shape = data[key].shape
    else:
        dims = data.dims
        shape = data.shape
    dims.remove(dim)
    shape.remove(data.sizes[dim])

    # Make empty data container
    empty = sc.DataArray(data=sc.zeros(dims=["tof"] + dims,
                                       shape=[nbins] + shape,
                                       variances=True,
                                       unit=sc.units.counts),
                         coords={"tof": tof_coord})

    for key in data.coords:
        if key != dim:
            empty.coords[key] = data.coords[key]

    if is_dataset:
        stitched = sc.Dataset()
        for i, (key, item) in enumerate(data.items()):
            for attr in item.attrs:
                if attr != dim:
                    empty.attrs[attr] = item.attrs[attr]
            stitched[key] = _stitch_item(item=item,
                                         dim=dim,
                                         frames=frames,
                                         target=empty.copy(),
                                         plot=(plot and i == 0))
    else:
        stitched = _stitch_item(item=data,
                                dim=dim,
                                frames=frames,
                                target=empty.copy(),
                                plot=plot)

    # Make sure to shift the position of the source to the midpoint between the
    # WFM choppers
    chopper_distances = data.meta["choppers"].value["distance"].data
    stitched.meta['source_position'] += sc.mean(
        sc.concatenate(chopper_distances["chopper", 0:2],
                       chopper_distances["chopper", 0:2], "none"))

    return stitched
