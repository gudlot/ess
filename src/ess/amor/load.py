# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
from typing import Optional
import warnings
import platform
from datetime import datetime
import scipp as sc
import scippneutron as scn
from orsopy import fileio
from .beamline import make_beamline
from ..logging import get_logger


def _tof_correction(data: sc.DataArray, dim: str = 'tof') -> sc.DataArray:
    """
    A correction for the presense of the chopper with respect to the "true" ToF.
    Also fold the two pulses.
    TODO: generalise mechanism to fold any number of pulses.
    """
    data.attrs['orso'].value.reduction.corrections += ['chopper ToF correction']
    tau = sc.to_unit(1 / (2 * data.coords['source_chopper'].value['frequency'].data),
                     data.coords[dim].unit)
    chopper_phase = data.coords['source_chopper'].value['phase'].data
    tof_offset = tau * chopper_phase / (180.0 * sc.units.deg)
    # Make 2 bins, one for each pulse
    edges = sc.concat([-tof_offset, tau - tof_offset, 2 * tau - tof_offset], dim)
    data = sc.bin(data, edges=[sc.to_unit(edges, data.coords[dim].unit)])
    # Make one offset for each bin
    offset = sc.concat([tof_offset, tof_offset - tau], dim)
    # Apply the offset on both bins
    data.bins.coords[dim] += offset
    # Rebin to exclude second (empty) pulse range
    return sc.bin(data, edges=[sc.concat([0. * sc.units.us, tau], dim)])


def load(filename,
         owner: fileio.base.Person,
         sample: fileio.data_source.Sample,
         creator: fileio.base.Person,
         reduction_script: str,
         beamline: Optional[dict] = None,
         disable_warnings: Optional[bool] = True) -> sc.DataArray:
    """
    Loader for a single Amor data file.

    :param filename: Path of the file to load.
    :param owner: the owner of the data set, i.e. the main proposer of the measurement.
    :param sample: A description of the sample.
    :param creator: The creator of the reduced data, the person responsible for the
        reduction process.
    :param reduction_script: The script or notebook used for reduction.
    :param beamline: A dict defining the beamline parameters.
    :param disable_warnings: Do not show warnings from file loading if `True`.
        Default is `True`.
    """
    get_logger('amor').info(
        "Loading '%s' as an Amor NeXus file",
        filename.filename if hasattr(filename, 'filename') else filename)
    if disable_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            data = scn.load_nexus(filename)
    else:
        data = scn.load_nexus(filename)

    # Convert tof nanoseconds to microseconds for convenience
    # TODO: is it safe to assume that the dtype of the binned wrapper coordinate is
    # the same as the dtype of the underlying event coordinate?
    data.bins.coords['tof'] = data.bins.coords['tof'].astype('float64', copy=False)
    data.coords['tof'] = data.coords['tof'].astype('float64', copy=False)
    data.bins.coords['tof'] = sc.to_unit(data.bins.coords['tof'], 'us', copy=False)
    data.coords['tof'] = sc.to_unit(data.coords['tof'], 'us', copy=False)

    # Add beamline parameters
    beamline = make_beamline() if beamline is None else beamline
    for key, value in beamline.items():
        data.coords[key] = value

    data.attrs['orso'] = sc.scalar(populate_orso())
    data.attrs['orso'].value.data_source.owner = owner
    data.attrs['orso'].value.data_source.experiment.title = data.attrs[
        'experiment_title'].value
    data.attrs['orso'].value.data_source.experiment.instrument = data.attrs[
        'instrument_name'].value
    data.attrs['orso'].value.data_source.experiment.start_date = datetime.strftime(
        datetime.strptime(data.attrs['start_time'].value[:-3],
                          '%Y-%m-%dT%H:%M:%S.%f'), '%Y-%m-%d')
    data.attrs['orso'].value.data_source.sample = sample
    data.attrs['orso'].value.data_source.measurement.data_files = [filename]
    data.attrs['orso'].value.reduction.creator = creator
    data.attrs['orso'].value.reduction.script = reduction_script

    # Perform tof correction and fold two pulses
    return _tof_correction(data)


def populate_orso() -> fileio.orso.Orso:
    """
    Populate the Orso object for metadata storage.

    :return: Populated Orso object.
    """
    orso = fileio.orso.Orso.empty()
    orso.data_source.experiment.probe = 'neutrons'
    orso.data_source.experiment.facility = 'Paul Scherrer Institut'
    orso.data_source.measurement.scheme = 'angle- and energy-dispersive'
    orso.reduction.software = fileio.reduction.Software(
        'scipp-ess', sc.__version__, platform.platform())
    orso.reduction.timestep = datetime.now()
    orso.reduction.corrections = []
    orso.reduction.computer = platform.node()
    orso.columns = [
        fileio.base.Column('Qz', '1/angstrom', 'wavevector transfer'),
        fileio.base.Column('R', None, 'reflectivity'),
        fileio.base.Column('sR', None, 'standard deivation of reflectivity'),
        fileio.base.Column('sQz',
                           '1/angstrom',
                           'standard deviation of wavevector transfer resolution'),
    ]
    return orso
