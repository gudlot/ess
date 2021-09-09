# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
from typing import Union


def _stitch_item(item: sc.DataArray, dim: str, frames: sc.Dataset, merge_frames: bool,
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
        for group in ["coords", "attrs"]:
            for key in getattr(item, group):
                if key != dim:
                    getattr(out, group)[key] = getattr(item, group)[key].copy()
    else:
        out = {}

    # Determine whether source_position is in coords or attrs
    coords_or_attrs = None
    for meta in ["coords", "attrs"]:
        if "source_position" in getattr(item, meta):
            coords_or_attrs = meta
    if coords_or_attrs is None:
        raise KeyError("'source_position' was not found in metadata.")

    for i in range(frames.sizes["frame"]):
        section = item[dim,
                       frames["time_min"].data["frame",
                                               i]:frames["time_max"].data["frame",
                                                                          i]].copy()
        section.coords['tof'] = section.meta[dim] - frames["time_correction"].data[
            "frame", i]
        del section.meta[dim]
        # TODO: when scipp 0.8 is released, rename_dims will create a new object.
        # section = section.rename_dims({dim: 'tof'})
        section.rename_dims({dim: 'tof'})
        print("===============================")
        print(section)
        print("###############################")
        print(section.events)
        if section.events is not None:
            section.events.coords['tof'] = section.events.meta[dim] - frames[
                "time_correction"].data["frame", i]
            del section.events.meta[dim]
            # TODO: when scipp 0.8 is released, rename_dims will create a new object.
            # section = section.rename_dims({dim: 'tof'})
            section.events.rename_dims({dim: 'tof'})
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(section.events)
        print("44444444444444444444444444444")
        print(section)

        if merge_frames:
            out += sc.rebin(section, 'tof', out.meta["tof"])
        else:
            getattr(section, coords_or_attrs
                    )['source_position'] = frames["wfm_chopper_mid_point"].data
            out[f"frame{i}"] = section

    # Note: we need to do the modification here because if not there is a coordinate
    # mismatch between `out` and `section`
    if merge_frames:
        getattr(
            out,
            coords_or_attrs)['source_position'] = frames["wfm_chopper_mid_point"].data

    return out


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
