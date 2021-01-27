import numpy as np
import scipp as sc
import matplotlib.pyplot as plt


def _stitch_item(item=None, dim=None, frames=None, target=None, plot=True):

    if plot:
        fig, ax = plt.subplots()

    for i in range(len(frames["left_edges"])):
        section = item[dim, frames["left_edges"][i] *
                       sc.units.us:frames["right_edges"][i] *
                       sc.units.us].copy()
        section.coords[dim] += frames["shifts"][i] * sc.units.us
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

    tof_coord = sc.Variable(
        ["tof"],
        unit=sc.units.us,
        values=np.linspace(frames["left_edges"][0] + frames["shifts"][0],
                           frames["right_edges"][-1] + frames["shifts"][-1],
                           nbins + 1))

    ind = data.dims.index(dim)
    dims = data.dims
    dims.remove(dim)
    shape = data.shape
    shape.remove(data.shape[ind])

    # Make empty data container
    empty = sc.DataArray(data=sc.zeros(dims=["tof"] + dims,
                                       shape=[nbins] + shape,
                                       variances=True,
                                       unit=sc.units.counts),
                         coords={"tof": tof_coord})

    for key in data.coords:
        if key != dim:
            empty.coords[key] = data.coords[key]

    if hasattr(data, "items"):
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
