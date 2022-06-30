# This file contains functions that will be deprecated. 
# NUrf scipp graveyard

def load_uv(name):
    """Loads the UV data from the corresponding entry in the LoKI.nxs filename.

    Parameters
    ----------
    name: str
        Filename, e.g. 066017.nxs

    Returns:
    ----------
    uv_dict: dict
        Dictionary that contains UV data signal (data) from the sample, the reference,
        and the dark measurement.
        Keys: sample, reference, dark

    """

    # load the nexus and extract the uv entry
    with snx.File(name) as f:
        uv = f["entry/instrument/uv"][()]

    # separation
    uv_dict = split_sample_dark_reference(uv)

    return uv_dict


def load_fluo(name):
    """Loads the data contained in the fluo entry of a LoKI.nxs file

    Parameters
    ----------
    name: str
        Filename, e.g. 066017.nxs

    Returns
    ----------
    fluo_dict: dict
        Dictionary of sc.DataArrays. Keys: data, reference, dark. Data contains the fluo signals of the sample.

    """

    with snx.File(name) as f:
        fluo = f["entry/instrument/fluorescence"][()]

    # separation
    fluo_dict = split_sample_dark_reference(fluo)

    return fluo_dict

    