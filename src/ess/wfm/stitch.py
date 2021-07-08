# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
import matplotlib.pyplot as plt


def _stitch_item(item=None, dim=None, frames=None, target=None, plot=True):

    if plot:
        fig, ax = plt.subplots()

    for i in range(frames.sizes["frame"]):
        section = item[dim, frames["left_edges"].data[
            "frame", i]:frames["right_edges"].data["frame", i]].copy()
        section.coords[dim] -= frames["shifts"].data["frame", i]
        section.rename_dims({dim: 'tof'})

        target += sc.rebin(section, 'tof', target.coords["tof"])
        if plot:
            section.name = "frame-{}".format(i)
            # TODO: we manually remove the 2d coord here, but this should be
            # made into a generic check
            del section.coords["position"]
            dims = section.dims
            dims.remove("tof")
            for dim_ in dims:
                section = sc.sum(section, dim_)
            section.plot(ax=ax, color="C{}".format(i))

    return target


def stitch(data=None, dim=None, frames=None, nbins=256, plot=False):

    tof_coord = sc.Variable(["tof"],
                            unit=sc.units.us,
                            values=np.linspace(
                                (frames["left_edges"]["frame", 0] -
                                 frames["shifts"]["frame", 0]).value,
                                (frames["right_edges"]["frame", -1] -
                                 frames["shifts"]["frame", -1]).value,
                                nbins + 1))

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

    return stitched
