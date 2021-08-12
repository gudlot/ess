import scipp as sc

def setup_offsets(sample, sample_trans, background, background_trans, direct_beam):
    # TODO: check this positions
    sample_pos_z_offset = 0.053 * sc.units.m
    monitor4_pos_z_offset = -6.719 * sc.units.m

    for item in [sample, sample_trans, background, background_trans, direct_beam]:
        # for item in [sample,sample_trans,background,background_trans,directbeam]:
        item.coords['sample_position'].fields.z += sample_pos_z_offset
        #item.coords['position'].fields.y += bench_pos_y_offset
        #TODO: this seems to be crititcal number of counts
        item.attrs['monitor4'].value.coords['position'].fields.z += monitor4_pos_z_offset


def setup_geometry(sample, background, direct_beam):

    for item in [sample, background, direct_beam]:
        item.coords['base_position'] = item.coords['position'].copy()
        x = -0.1057 * sc.units.m
        y = 0.082735 * sc.units.m
        z = 0.0 * sc.units.m
        offset = sc.geometry.position(x, y, z)
        item.coords['position'] = item.coords['base_position'] + offset
