import scipp as sc


def setup_offsets(
    data_set,
    sample_pos_z_offset,
    bench_pos_y_offset,
    monitor4_pos_z_offset,
):
    """
    Transformin coordinates according to instrument setup
    """
    data_set.coords["sample_position"].fields.z += sample_pos_z_offset
    data_set.coords["position"].fields.y += bench_pos_y_offset
    for item in data_set.keys():
        data_set[item].attrs["monitor4"].value.coords["position"].fields.z += monitor4_pos_z_offset


def setup_geometry(data_set, x_offset, y_offset, z_offset):
    """
    Transforming coordinates according to beam center positons
    """
    data_set.coords['position'].fields.x += x_offset
    data_set.coords['position'].fields.y += y_offset
    data_set.coords['position'].fields.z += z_offset