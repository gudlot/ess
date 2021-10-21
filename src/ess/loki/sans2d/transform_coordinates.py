import scipp as sc


def setup_offsets(
    sample,
    sample_trans,
    background,
    background_trans,
    direct_beam,
    sample_pos_z_offset,
    bench_pos_y_offset,
    monitor4_pos_z_offset,
):
    """
    Transformin coordinates according to instrument setup
    """
    for item in [sample, sample_trans, background, background_trans, direct_beam]:
        # for item in [sample,sample_trans,background,background_trans,directbeam]:
        item.coords["sample_position"].fields.z += sample_pos_z_offset
        item.coords["position"].fields.y += bench_pos_y_offset
        # TODO: this seems to be crititcal number of counts
        item.attrs["monitor4"].value.coords[
            "position"
        ].fields.z += monitor4_pos_z_offset


def setup_geometry(sample, background, x_offset, y_offset, z_offset):
    """
    Transforming coordinates according to beam center positons
    """
    for item in [sample, background]:
        item.coords["base_position"] = item.coords["position"].copy()
        offset = sc.geometry.position(x_offset, y_offset, z_offset)
        item.coords["position"] = item.coords["base_position"] + offset
