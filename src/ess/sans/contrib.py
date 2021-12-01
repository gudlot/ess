# Generic helpers that may end up as contributions to scipp if cleaned up
import scipp as sc


def midpoints(var, dim):
    """
    Utility function for finding middle points of variable
    """
    return 0.5 * (var[dim, 1:] + var[dim, :-1])


def to_bin_centers(d, dim):
    """
    Utility function for setting centers of bins
    """
    d.coords[dim] = midpoints(d.coords[dim], dim)

def to_bin_edges(d, dim):
    """
    Utility function for setting bin edges
    """
    centers = d.coords[dim]
    del d.coords[dim]
    first = 1.5 * centers[dim, 0] - 0.5 * centers[dim, 1]
    last = 1.5 * centers[dim, -1] - 0.5 * centers[dim, -2]
    bulk = midpoints(centers, dim)
    d.coords[dim] = sc.concat([first, bulk, last], dim)


def map_to_bins(data, dim, edges):
    """
    Utility function for binning data according to preset edges
    """
    data = data.copy()
    to_bin_edges(data, dim)
    bin_width = data.coords[dim][dim, 1:] - data.coords[dim][dim, :-1]
    bin_width.unit = sc.units.one
    data *= bin_width
    data = sc.rebin(data, dim, edges)
    bin_width = edges[dim, 1:] - edges[dim, :-1]
    bin_width.unit = sc.units.one
    data /= bin_width
    return data


