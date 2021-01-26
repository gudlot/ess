import numpy as np
import scipp as sc


def _stitch_item(item=None, dim=None, frames=None, target=None, plot=True):


    # corrected = []
    # stitched
    for i in range(len(frames["left_edges"])):
        section = item[dim,
                             frames["left_edges"][i]*sc.units.us:frames["right_edges"][i]*sc.units.us].copy()
        section.coords[dim] += frames["shifts"][i]*sc.units.us
        section.rename_dims({dim: 'tof'})

        target += sc.rebin(section, 'tof', target.coords["tof"])

        # corrected.append(section)

        #             # Sum counts from different frames
        #     for key in sections:
        #         for sec in sections[key]:
        #             stitched[key] += sc.rebin(sec, 'tof', stitched.coords["tof"])


    # empty


    return target



def stitch(data=None, dim=None, frames=None, nbins=None):

    tof_coord = sc.Variable(["tof"],
                                 unit=sc.units.us,
                                 values=np.linspace(frames["left_edges"][0] + 
                                    frames["shifts"][0],
                                    frames["right_edges"][-1] + 
                                    frames["shifts"][-1], nbins + 1))

    ind = data.dims.index(dim)
    dims = data.dims
    dims.remove(dim)
    shape = data.shape
    shape.remove(data.shape[ind])

    # Make empty data container
    empty = sc.DataArray(data=sc.zeros(dims=["tof"] + dims, shape=[nbins] + shape,
                                     variances=True, unit=sc.units.counts),
                         coords={"tof": tof_coord})

    for key in data.coords:
        if key != dim:
            empty.coords[key] = data.coords[key]

    if hasattr(data, "items"):
        sections = {}
        stitched = sc.Dataset()
        for key, item in data.items():
            # stitched[key] = empty
            stitched[key] = _stitch_item(item=item, dim=dim, frames=frames, target=empty.copy())
    # else:






# plot({"frame{}".format(i): sc.sum(sc.sum(sections["reference"][i], 'x'), 'y')
#       for i in range(len(sections["reference"]))})



    # # stitched.coords["tof"] = sc.Variable(["tof"],
    # #                              unit=sc.units.us,
    # #                              values=np.linspace(9.0e3, 5.0e4, ntof + 1))
    # # for key in ds.coords:
    # #     if key != "t":
    # #         stitched.coords[key] = ds.coords[key]
    # for key in ds:
    #     stitched[key] = sc.zeros(dims=["tof", "y", "x"], shape=[ntof] + ds.coords["position"].shape,
    #                                  variances=True, unit=sc.units.counts)
    # # Sum counts from different frames
    # for key in sections:
    #     for sec in sections[key]:
    #         stitched[key] += sc.rebin(sec, 'tof', stitched.coords["tof"])
    # stitched


    # for key in ds.coords:
    #     if key != "t":
    #         stitched.coords[key] = ds.coords[key]



    return stitched
