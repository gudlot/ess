import csv
import glob
import os
import re

import numpy as np
import scipp as sc
import tifffile
from astropy.io import fits


def read_x_values(tof_file, delimiter=None, skiprows=0, usecols=None):
    """
    Reads the TOF values from the CSV into a list.
    If usecols is defined, we use the column requested by the user.
    If not, as there may be more than one column in the file (typically the
    counts are also stored in the file alongside the TOF bins), we search for
    the first column with monotonically increasing values.
    """
    data = np.loadtxt(tof_file,
                      delimiter=delimiter,
                      skiprows=skiprows,
                      usecols=usecols)
    if (usecols is not None) or (data.ndim == 1):
        return data

    # Search for the first column with monotonically increasing values
    for i in range(data.shape[1]):
        if np.all(data[1:, i] > data[:-1, i], axis=0):
            return data[:, i]

    raise RuntimeError("No column with monotonically increasing values was "
                       "found in file " + tof_file)


def _load_images(image_dir, extension, loader):
    if not os.path.isdir(image_dir):
        raise ValueError(image_dir + " is not directory")
    stack = []
    path_length = len(image_dir) + 1
    filenames = glob.glob(image_dir + f"/*.{extension}")
    # Sort the filenames by converting the digits in the strings to integers
    filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
    nfiles = len(filenames)
    count = 0
    print(f"Loading {nfiles} files from '{image_dir}'")
    for filename in filenames:
        count += 1
        print('\r{0}: Image {1}, of {2}'.format(filename[path_length:], count,
                                                nfiles),
              end="")
        img = loader(os.path.join(image_dir, filename))
        stack.append(np.flipud(img.data))

    print()  # Print a newline to separate each load message

    return np.array(stack)


def _load_fits(fits_dir):
    def loader(f):
        data = None
        handle = fits.open(f, mode='readonly')
        data = handle[0].data.copy()
        handle.close()
        return data

    return _load_images(fits_dir, 'fits', loader)


def _load_tiffs(tiff_dir):
    return _load_images(tiff_dir, 'tiff', lambda f: tifffile.imread(f))


def export_tiff_stack(dataset, key, base_name, output_dir, x_len, y_len,
                      tof_values):
    to_save = dataset[key]

    num_bins = 1 if len(to_save.shape) == 1 else to_save.shape[0]
    stack_data = np.reshape(to_save.values, (x_len, y_len, num_bins))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Writing tiffs
    for i in range(stack_data.shape[2]):
        tifffile.imsave(
            os.path.join(output_dir, '{:s}_{:04d}.tiff'.format(base_name, i)),
            stack_data[:, :, i].astype(np.float32))
    print('Saved {:s}_{:04d}.tiff stack.'.format(base_name, 0))

    # Write out tofs as CSV
    tof_vals = [tof_values[0], tof_values[-1]] if num_bins == 1 else tof_values

    with open(os.path.join(output_dir, 'tof_of_tiff_{}.txt'.format(base_name)),
              'w') as tofs:
        writer = csv.writer(tofs, delimiter='\t')
        writer.writerow(['tiff_bin_nr', 'tof'])
        tofs = tof_vals
        tof_data = list(zip(list(range(len(tofs))), tofs))
        writer.writerows(tof_data)
    print('Saved tof_of_tiff_{}.txt.'.format(base_name))


def _image_to_variable(image_dir,
                       loader,
                       dtype=np.float64,
                       with_variances=True):
    """
    Loads all images from the directory into a scipp Variable.
    """
    stack = loader(image_dir)
    data = stack.astype(dtype).reshape(stack.shape[0],
                                       stack.shape[1] * stack.shape[2])
    if with_variances:
        return sc.Variable(["t", "spectrum"], values=data, variances=data)
    else:
        return sc.Variable(["t", "spectrum"], values=data)


def tiffs_to_variable(tiff_dir, dtype=np.float64, with_variances=True):
    """
    Loads all tiff images from the directory into a scipp Variable.
    """
    return _image_to_variable(tiff_dir, _load_tiffs, dtype, with_variances)


def fits_to_variable(fits_dir, dtype=np.float64, with_variances=True):
    """
    Loads all fits images from the directory into a scipp Variable.
    """
    return _image_to_variable(fits_dir, _load_fits, dtype, with_variances)


def make_detector_groups(nx_original, ny_original, nx_target, ny_target):
    element_width_x = nx_original // nx_target
    element_width_y = ny_original // ny_target

    x = sc.Variable(dims=['x'],
                    values=np.arange(nx_original) // element_width_x)
    y = sc.Variable(dims=['y'],
                    values=np.arange(ny_original) // element_width_y)
    grid = x + nx_target * y
    return sc.Variable(["spectrum"], values=np.ravel(grid.values))
