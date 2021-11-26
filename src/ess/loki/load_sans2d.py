import scipp as sc
import scippneutron as scn


def load_isis(filename, spectrum_size, tof_bins):
    """
    Loading wrapper for ISIS files
    """
    events = scn.load(filename=filename, mantid_args={"LoadMonitors": True})
    #TODO: Remove histograming line once it is confirmed it works without
    #return sc.histogram(events["spectrum", :spectrum_size], tof_bins)
    return sc.bin(events["spectrum", :spectrum_size], edges=[tof_bins])

def load_rkh_wav(filename):
    """
    Loading wrapper for RKH files
    """
    return scn.load(
        filename=filename,
        mantid_alg="LoadRKH",
        mantid_args={"FirstColumnValue": "Wavelength"},
    )


def load_rkh_q(filename):
    """
    Loading wrapper for RKH files
    """
    return scn.load(
        filename=filename,
        mantid_alg="LoadRKH",
        mantid_args={"FirstColumnValue": "MomentumTransfer"},
    )


def load_mask(idf_filename, mask_file):
    """
    Loading individual mask file
    """
    return scn.load(
        filename=idf_filename,
        mantid_alg="LoadMask",
        mantid_args={"InputFile": mask_file},
    )


def load_and_apply_masks(
    idf_filename,
    mask_files,
    data_set,
    spectrum_size,
):
    """
    Loading masks files from the list and add them to sample and background
    """
    for i, mask_file in enumerate(mask_files):
        mask_xml = load_mask(idf_filename, mask_file)
        for item in data_set.keys():
            data_set[item].masks[f"mask_{i}_xml"] = mask_xml["spectrum", :spectrum_size].data

def apply_tof_mask(data_set):
    """
    TOF mask for SANS2D data
    """
    tof = data_set.coords["tof"]
    for item in data_set.keys():
        data_set[item].masks["bins"] = sc.less(tof["tof", 1:], 8000.0 * sc.units.us) | (
            sc.greater(tof["tof", :-1], 13000.0 * sc.units.us)
            & sc.less(tof["tof", 1:], 15750.0 * sc.units.us))
