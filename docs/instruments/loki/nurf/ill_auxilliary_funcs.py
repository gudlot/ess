

def complete_fname(scan_numbers):
    """Converts a list of input numbers to a filename uses at ILL.

    Parameters
    ----------
    scan_numbers: list of int or a single int
        List of filenumbers or one filenumnber.

    Returns:
    ----------
    flist_num: list of str or  one  str
        List of filenames following ILL style or string following ILL style.

    """
    if isinstance(scan_numbers, int):
        flist_num = f"{str(scan_numbers).zfill(6)}.nxs"

    if isinstance(scan_numbers, list):
        # convert a list of input numbers to real filename
        flist_num = [str(i).zfill(6) + ".nxs" for i in scan_numbers]

    return flist_num