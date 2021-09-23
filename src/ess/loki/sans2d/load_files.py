import scipp as sc
import scippneutron as scn

def load_isis(filename, spectrum_size, tof_bins):
    events = scn.load(filename=filename, mantid_args={'LoadMonitors':True})
    return sc.histogram(events['spectrum',:spectrum_size], tof_bins)

def load_rkh_wav(filename):
    return scn.load(
           filename=filename,
           mantid_alg='LoadRKH',
           mantid_args={'FirstColumnValue':'Wavelength'})

def load_rkh_q(filename):
    return scn.load(
           filename=filename,
           mantid_alg='LoadRKH',
           mantid_args={'FirstColumnValue':'MomentumTransfer'})

def load_mask(idf_filename, mask_file):

    return scn.load(filename=idf_filename, mantid_alg='LoadMask', mantid_args={'InputFile': mask_file})

def load_masks(idf_filename, mask_files, sample, background, spectrum_size):

    mask_1_xml = load_mask(idf_filename, mask_files[0])
    mask_2_xml = load_mask(idf_filename, mask_files[1])
    mask_3_xml = load_mask(idf_filename, mask_files[2])
    mask_4_xml = load_mask(idf_filename, mask_files[3])
    mask_5_xml = load_mask(idf_filename, mask_files[4])
    mask_6_xml = load_mask(idf_filename, mask_files[5])
    mask_7_xml = load_mask(idf_filename, mask_files[6])
    mask_8_xml = load_mask(idf_filename, mask_files[7])
    mask_9_xml = load_mask(idf_filename, mask_files[8])

    sample.masks['mask_1_xml'] = mask_1_xml['spectrum',:spectrum_size].data
    sample.masks['mask_2_xml'] = mask_2_xml['spectrum',:spectrum_size].data
    sample.masks['mask_3_xml'] = mask_3_xml['spectrum',:spectrum_size].data
    sample.masks['mask_4_xml'] = mask_4_xml['spectrum',:spectrum_size].data
    sample.masks['mask_5_xml'] = mask_5_xml['spectrum',:spectrum_size].data
    sample.masks['mask_6_xml'] = mask_6_xml['spectrum',:spectrum_size].data
    sample.masks['mask_7_xml'] = mask_7_xml['spectrum',:spectrum_size].data
    sample.masks['mask_8_xml'] = mask_8_xml['spectrum',:spectrum_size].data
    sample.masks['mask_9_xml'] = mask_9_xml['spectrum',:spectrum_size].data

    background.masks['mask_1_xml'] = mask_1_xml['spectrum',:spectrum_size].data
    background.masks['mask_2_xml'] = mask_2_xml['spectrum',:spectrum_size].data
    background.masks['mask_3_xml'] = mask_3_xml['spectrum',:spectrum_size].data
    background.masks['mask_4_xml'] = mask_4_xml['spectrum',:spectrum_size].data
    background.masks['mask_5_xml'] = mask_5_xml['spectrum',:spectrum_size].data
    background.masks['mask_6_xml'] = mask_6_xml['spectrum',:spectrum_size].data
    background.masks['mask_7_xml'] = mask_7_xml['spectrum',:spectrum_size].data
    background.masks['mask_8_xml'] = mask_8_xml['spectrum',:spectrum_size].data
    background.masks['mask_9_xml'] = mask_9_xml['spectrum',:spectrum_size].data