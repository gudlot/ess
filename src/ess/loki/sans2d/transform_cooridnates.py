import scipp as sc


def transform_position(sample, background, directbeam):
    sample_pos_offset = sc.vector(value=[0.0, 0.0, 0.053], unit=sc.units.m)
    bench_pos_offset = sc.vector(value=[0.0, 0.001, 0.0], unit=sc.units.m)
    monitor4_pos_offset = sc.vector(value=[0.0, 0.00, -6.719], unit=sc.units.m)
    #sample_pos_z_offset = 0.053 * sc.Unit('m')
    #bench_pos_y_offset = 0.001 * sc.Unit('m')
    #monitor4_pos_z_offset = -6.719 * sc.Unit('m')

    for item in [sample,background, directbeam]:
    #for item in [sample,sample_trans,background,background_trans,directbeam]:
        item.coords['sample_position'] += sample_pos_offset
        item.attrs['monitor2'].value.coords['sample_position'] += sample_pos_offset
        item.attrs['monitor4'].value.coords['sample_position'] += sample_pos_offset
        item.coords['position'] += bench_pos_offset
        item.attrs['monitor4'].value.coords['position'] += monitor4_pos_offset

def transform_geometry(sample, background):
    # 105.700 -82.735
    x = -0.082735 * sc.units.m
    y = 0.1057 * sc.units.m
    # Alternative - in Mantid user file?
    # x = -0.085 * sc.units.m
    # y = 0.1419 * sc.units.m
    z = 0.0 * sc.units.m
    offset = sc.geometry.position(x, y, z)
    sample.coords['position'] = sample.coords['base_position'] + offset
    background.coords['position'] = background.coords['base_position'] + offset