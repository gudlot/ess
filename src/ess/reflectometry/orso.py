# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# flake8: noqa: E501
"""
This is the implementation of the ORSO header information.
"""

# author: Andrew R. McCluskey (arm61)

import yaml
import socket
import datetime
import pathlib
import getpass
from ess import __version__ as VERSION

OSRO_VERSION = 0.1


def noop(self, *args, **kw):
    pass


yaml.emitter.Emitter.process_tag = noop


def _repr(class_to_represent):
    """
    The representation object for all the Header sub-classes. This returns a string in a yaml format which will be ORSO compatible.

    Args:
        class_to_represent (:py:class:`object`): The class to be represented.

    Returns:
        (:py:attr:`str`): A string representation.
    """
    return yaml.dump(class_to_represent, sort_keys=False)


class Header:
    """
    The super class for all of the __repr__ items in the orso module
    """
    def __repr__(self):
        """
        The string representation for the Header class objects.
        """
        return _repr(self)


class Orso:
    """
    The class for the Orso header object.

    Args:
        creator (:py:class:`orso.Creator`): Information about the creation of the reduction.
        data_source (:py:class:`orso.DataSource`): Details of the data being reduced.
        reduction (:py:class:`orso.Reduction`): Information about the reduction that is performed.
        columns (:py:attr:`list` of :py:class:`orso.Column`): A list of the different columns persent in the data.
    """
    def __init__(self, creator, data_source, reduction, columns):
        self.creator = creator
        self.data_source = data_source
        self.reduction = reduction
        self.columns = columns

    def __repr__(self):
        """
        To ensure the prescence of the ORSO top line, the `orso.Orso` class has a slightly different :code:`__repr__`.
        """
        return f'# ORSO reflectivity data file | {OSRO_VERSION:.1f} standard | YAML encoding | https://reflectometry.org\n' + _repr(
            self)


class Creator(Header):
    """
    The information about who and when the reduced data was created, ie. when the reduction was performed and by whom.

    Args:
        name (:py:attr:`str`, optional): The name of the person that performed the reduction, can also include an email address. Optional, defaults to the machine username.
        affiliation (:py:attr:`str`, optional): The affiliation of the person that performed the reduction. Optional, defaults to :code:`None`.
        time (:py:attr:`str`, optional): The time that the reduction was performed, in the format :code:`"%Y-%m-%dT%H:%M:%S"`. Optional, defaults to the current time.
        system (:py:attr:`str`, optional): The machine that the reduction was performed on. Optional, defaults to the machine's host name.
    """
    def __init__(self, name=None, affiliation=None, time=None, system=None):
        self.name = name
        if name is None:
            self.name = getpass.getuser()
        if affiliation is not None:
            self.affiliation = affiliation
        self.time = time
        if time is None:
            self.time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.system = system
        if system is None:
            self.system = socket.gethostname()


class Sample(Header):
    """
    Sample information.

    Args:
        name (:py:attr:`str`, optional): A identifiable description for the sample. Optional, defaults to :code:`'Sample'`.
    """
    def __init__(self, name='Sample'):
        self.name = name


class ValueScalar(Header):
    """
    A single value with an unit.

    Args:
        magnitude (:py:attr:`float`): The value.
        unit (:py:attr:`str`, optional): The unit. Optional, defaults to :code:`'dimensionless'`.
    """
    def __init__(self, magnitude, unit='dimensionless'):
        self.magnitude = magnitude
        if unit == '\xC5':
            unit = 'angstrom'
        self.unit = unit


class ValueRange(Header):
    """
    A range with an upper and lower limit and a unit.

    Args:
        min (:py:attr:`float`): The minimum value.
        max (:py:attr:`float`): The maximum value.
        unit (:py:attr:`str`, optional): The unit. Optional, defaults to :code:`'dimensionless'`.
    """
    def __init__(self, min, max, unit='dimensionless'):
        self.min = min
        self.max = max
        if unit == '\xC5':
            unit = 'angstrom'
        self.unit = unit


class Measurement(Header):
    """
    Details of the measurement that is performed.

    Args:
        scheme (:py:attr:`str`, optional): The measurement scheme (ie. :code:`'energy-dispersive'`). Optional, defaults to :code:`'None'`.
        omega (:py:class:`orso.ValueScalar` or :py:class:`orso.ValueRange`, optional): The incident angle value or range. Optional, defaults to :code:`'None'`.
        wavelength (:py:class:`orso.ValueScalar` or :py:class:`orso.ValueRange`, optional): The measured wavelength value or range. Optional, defaults to :code:`'None'`.
        polarisation (:py:attr:`str`, optional): The polarisation present, typically as a :code:`'+'` or :code:`'-'` or combination. Optional, defaults to :code:`'None'`.
    """
    def __init__(self, scheme=None, omega=None, wavelength=None, polarisation=None):
        if scheme is not None:
            self.scheme = scheme
        if omega is not None:
            self.omega = omega
        if wavelength is not None:
            self.wavelength = wavelength
        if polarisation is not None:
            self.polarisation = polarisation


class Experiment(Header):
    """
    Experimental details.

    Args:
        instrument (:py:attr:`str`, optional): The name of the instrument. Optional, defaults to :code:`'None'`.
        probe (:py:attr:`str`, optional): The name of the probing radiation. Optional, defaults to :code:`'None'`.
        sample (:py:class:`orso.Sample`, optional): A description of the sample. Optional, defaults to :code:`'None'`.
    """
    def __init__(self, instrument=None, probe=None, sample=None):
        if instrument is not None:
            self.instrument = instrument
        if probe is not None:
            self.probe = probe
        if sample is not None:
            self.sample = sample


class DataSource(Header):
    """
    Details of where and who the data came from.

    Args:
        owner (:py:attr:`str`, optional): The name (and affiliation/email address) of the owner of the data. Optional, defaults to :code:`'None'`.
        facility (:py:attr:`str`, optional): The name of the facility the data was generated at. Optional, defaults to :code:`'None'`.
        experiment_id (:py:attr:`str`, optional): An identifier for the experiment (ie. proposal ID). Optional, defaults to :code:`'None'`.
        experiment_date (:py:attr:`str`, optional): The date or date range that the experiment was conducted on, in the format :code:`"%Y-%m-%d"`. Optional, defaults to :code:`'None'`.
        title (:py:attr:`str`, optional): A name of the data source. Optional, defaults to :code:`'None'`.
        experiment (:py:class:`orso.Experiment`, optional): Information about the experimental setup. Optional, defaults to :code:`'None'`.
        measurement (:py:class:`orso.Measurement`, optional): Details of the measurement scheme. Optional, defaults to :code:`'None'`.
    """
    def __init__(self,
                 owner=None,
                 facility=None,
                 experiment_id=None,
                 experiment_date=None,
                 title=None,
                 experiment=None,
                 measurement=None):
        if owner is not None:
            self.owner = owner
        if facility is not None:
            self.facility = facility
        if experiment_id is not None:
            self.experiment_id = experiment_id
        self.experiment_date = experiment_date
        if experiment_date is None:
            self.experiment_date = datetime.datetime.now().strftime("%Y-%m-%d")
        if title is not None:
            self.title = title
        if experiment is not None:
            self.experiment = experiment
        if measurement is not None:
            self.measurement = measurement


class File(Header):
    """
    Information for a given file.

    Attributes:
        creation (:py:attr:`str`): The date and time of the file creation.

    Args:
        file (:py:attr:`str`): The file name/path.
    """
    def __init__(self, file):
        self.file = file
        fpath = pathlib.Path(file)
        if not fpath.exists():
            raise FileNotFoundError(f'The file {file} could not be found.')
        else:
            self.creation = datetime.datetime.fromtimestamp(
                fpath.stat().st_ctime).strftime("%Y-%m-%dT%H:%M:%S")


class Files(Header):
    """
    Information on data files and associated reference files.

    Args:
        data_files (:py:attr:`list` of :py:class:`orso.File`): The experimental data files.
        reference_files (:py:attr:`list` of :py:class:`orso.File`, optional): The reference files. Optional, defaults to :code:`'None'`.
    """
    def __init__(self, data_files, reference_files=None):
        self.data_files = data_files
        if reference_files is not None:
            self.reference_files = reference_files


class Reduction(Header):
    """
    Details of the reduction processes.

    Args:
        script (:py:attr:`str`, optional): The file name/path for the reduction script or notebook. Optional, defaults to :code:`'None'`.
        input_files (:py:class:`orso.Files`, optional): The input files for the reduction. Optional defaults to :code:`'None'`.
        comments (:py:attr:`str`, optional): An additional comment on the reduction. Optional, defaults to :code:`'None'`.
    """
    def __init__(self, script=None, input_files=None, comment=None):
        self.software = f'ess-{VERSION}'
        if script is not None:
            self.script = script
        if input_files is not None:
            self.input_files = input_files
        if comment is not None:
            self.comment = comment


class Column:
    """
    Information on a data column.

    Args:
        quantity (:py:attr:`str`): The name of the column.
        unit (:py:attr:`str`, optional): The unit. Optional, defaults to :code:`'dimensionless'`.
        comments (:py:attr:`str`, optional): An additional comment on the column (ie. the definition for an uncertainty column). Optional, defaults to :code:`'None'`.
    """
    def __init__(self, quantity, unit='dimensionless', comment=None):
        self.quantity = quantity
        self.unit = unit
        if comment is not None:
            self.comment = comment
