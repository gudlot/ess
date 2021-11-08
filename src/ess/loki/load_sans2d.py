import scipp as sc
import scippneutron as scn


def load_isis(filename, spectrum_size, tof_bins):
    """
    Loading wrapper for ISIS files
    """
    events = scn.load(filename=filename, mantid_args={"LoadMonitors": True})
    return sc.histogram(events["spectrum", :spectrum_size], tof_bins)


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


def load_and_apply_masks(idf_filename, mask_files, sample, background, spectrum_size):
    """
    Loading masks files from the list and add them to sample and background
    """
    for i, mask_file in enumerate(mask_files):
        mask_xml = load_mask(idf_filename, mask_file)
        sample.masks[f"mask_{i}_xml"] = mask_xml["spectrum", :spectrum_size].data
        background.masks[f"mask_{i}_xml"] = mask_xml["spectrum", :spectrum_size].data
