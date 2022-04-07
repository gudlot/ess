# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import scipp as sc


def normalise_by_supermirror(sample: sc.DataArray,
                             supermirror: sc.DataArray) -> sc.DataArray:
    """
    Normalise the sample measurement by the (ideally calibrated) supermirror.

    :param sample: Sample measurement with coords of 'Q' and 'detector_id'.
    :param supermirror: Supermirror measurement with coords of 'Q' and 'detector_id',
        ideally calibrated.

    :return: Normalised sample.
    """
    normalised = sample / supermirror
    normalised.masks['no_reference_neutrons'] = (supermirror == sc.scalar(0)).data
    try:
        normalised.attrs['orso'] = sample.attrs['orso']
        normalised.attrs['orso'].value.reduction.corrections = list(
            set(sample.attrs['orso'].value.reduction.corrections +
                supermirror.attrs['orso'].value.reduction.corrections))
        normalised.attrs[
            'orso'].value.data_source.measurement.reference = supermirror.attrs[
                'orso'].value.data_source.measurement.data_files
    except KeyError:
        raise UserWarning("For metadata to be logged in the data array, "
                          "it is necessary to install the orsopy package.")
    return normalised
