import scipp as sc

def setup_offsets(sample, sample_trans, background, background_trans, direct_beam):
    # TODO: check this positions
    sample_pos_z_offset = 0.053 * sc.units.m
    bench_pos_y_offset = 0.001 * sc.units.m
    monitor4_pos_z_offset = -6.719 * sc.units.m

    for item in [sample, sample_trans, background, background_trans, direct_beam]:
        # for item in [sample,sample_trans,background,background_trans,directbeam]:
        item.coords['sample_position'].fields.z += sample_pos_z_offset
        item.coords['position'].fields.y += bench_pos_y_offset
        #TODO: this seems to be crititcal number of counts
        item.attrs['monitor4'].value.coords['position'].fields.z += monitor4_pos_z_offset


def setup_geometry(sample, background, x, y, z):

    for item in [sample, background]:
        item.coords['base_position'] = item.coords['position'].copy()
        x_offset = x * sc.units.m
        y_offset = y * sc.units.m
        z_offset = z * sc.units.m
        offset = sc.geometry.position(x_offset, y_offset, z_offset)
        item.coords['position'] = item.coords['base_position'] + offset
