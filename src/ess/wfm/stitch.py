# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
from typing import Union


def _populate_coords(old_item, new_item, dim):
    for group in ["coords", "attrs"]:
        for key in getattr(old_item, group):
            if key != dim:
                getattr(new_item, group)[key] = getattr(old_item, group)[key].copy()


def _update_source_position(item, frames):
    coords_or_attrs = None
    for meta in ["coords", "attrs"]:
        if "source_position" in getattr(item, meta):
            coords_or_attrs = meta
    if coords_or_attrs is None:
        raise KeyError("'source_position' was not found in metadata.")
    getattr(item,
            coords_or_attrs)['source_position'] = frames["wfm_chopper_mid_point"].data


def _stitch_dense_data(item: sc.DataArray, dim: str, frames: sc.Dataset,
                       merge_frames: bool,
                       bins: Union[int, sc.Variable]) -> Union[sc.DataArray, dict]:

    if merge_frames:
        # Make empty data container
        if isinstance(bins, int):
            tof_coord = sc.linspace(
                dim="tof",
                start=(frames["time_min"]["frame", 0] -
                       frames["time_correction"]["frame", 0]).value,
                stop=(frames["time_max"]["frame", -1] -
                      frames["time_correction"]["frame", -1]).value,
                num=bins + 1,
                unit=frames["time_min"].unit,
            )
        else:
            tof_coord = bins

        dims = []
        shape = []
        for dim_ in item.dims:
            if dim_ != dim:
                dims.append(dim_)
                shape.append(item.sizes[dim_])
            else:
                dims.append('tof')
                shape.append(tof_coord.sizes['tof'] - 1)

        out = sc.DataArray(data=sc.zeros(dims=dims,
                                         shape=shape,
                                         variances=item.variances is not None,
                                         unit=item.unit),
                           coords={"tof": tof_coord})
        _populate_coords(item, out, dim)
    else:
        out = {}

    for i in range(frames.sizes["frame"]):
        section = item[dim,
                       frames["time_min"].data["frame",
                                               i]:frames["time_max"].data["frame",
                                                                          i]].copy()
        section.coords['tof'] = section.meta[dim] - frames["time_correction"].data[
            "frame", i]
        del section.meta[dim]
        section.rename_dims({dim: 'tof'})

        if merge_frames:
            out += sc.rebin(section, 'tof', out.meta["tof"])
        else:
            _update_source_position(section, frames)
            out[f"frame{i}"] = section

    # Note: we need to do the modification here because if not there is a coordinate
    # mismatch between `out` and `section`
    if merge_frames:
        _update_source_position(out, frames)

    return out


def _stitch_event_data(item: sc.DataArray, dim: str, frames: sc.Dataset,
                       merge_frames: bool,
                       bins: Union[int, sc.Variable]) -> Union[sc.DataArray, dict]:

    if merge_frames:
        out = None
    else:
        out = {}

    for i in range(frames.sizes["frame"]):
        piece = sc.bin(item,
                       edges=[
                           sc.concatenate(frames["time_min"].data["frame", i],
                                          frames["time_max"].data["frame", i], 'time')
                       ])
        piece.bins.coords[
            'tof'] = piece.bins.meta[dim] - frames["time_correction"].data["frame", i]
        piece.coords['tof'] = piece.meta['time'] - frames["time_correction"].data[
            "frame", i]
        del piece.meta['time']
        del piece.bins.meta['time']
        section = piece.events.copy()
        section.rename_dims({'time': 'tof'})

        if merge_frames:
            if out is None:
                out = section
            else:
                out = sc.concatenate(out, section, 'tof')
        else:
            _populate_coords(item, section, dim)
            _update_source_position(section, frames)
            out[f"frame{i}"] = section

    # Note: we need to do the modification here because if not there is a coordinate
    # mismatch between `out` and `section`
    if merge_frames:
        edges = sc.array(dims=['tof'],
                         values=[(frames["time_min"]["frame", 0] -
                                  frames["time_correction"]["frame", 0]).value,
                                 (frames["time_max"]["frame", -1] -
                                  frames["time_correction"]["frame", -1]).value],
                         unit=frames["time_min"].unit)
        out = sc.bin(out, edges=[edges])
        _populate_coords(item, out, dim)
        _update_source_position(out, frames)

    return out


def _stitch_item(item: sc.DataArray, **kwargs) -> Union[sc.DataArray, dict]:
    if item.bins is not None:
        return _stitch_event_data(item, **kwargs)
    else:
        return _stitch_dense_data(item, **kwargs)


def stitch(
        data: Union[sc.DataArray, sc.Dataset],
        dim: str,
        frames: sc.Dataset,
        merge_frames: bool = True,
        bins: Union[int, sc.Variable] = 256) -> Union[sc.DataArray, sc.Dataset, dict]:
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
        frames["time_min"] = sc.mean(frames["time_min"], _dim)
        frames["time_max"] = sc.mean(frames["time_max"], _dim)

    if isinstance(data, sc.Dataset):
        if merge_frames:
            stitched = sc.Dataset()
        else:
            stitched = {}
        for i, (key, item) in enumerate(data.items()):
            stitched[key] = _stitch_item(item=item,
                                         dim=dim,
                                         frames=frames,
                                         merge_frames=merge_frames,
                                         bins=bins)
    else:
        stitched = _stitch_item(item=data,
                                dim=dim,
                                frames=frames,
                                merge_frames=merge_frames,
                                bins=bins)

    return stitched
