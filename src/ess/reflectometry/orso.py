# flake8: noqa: E501
"""
This is the implementation of the ORSO header information.
"""
import yaml
import socket
import datetime
import pathlib
import getpass

OSRO_VERSION = 0.1
VERSION = '0.0.1'


def noop(self, *args, **kw):
    pass


yaml.emitter.Emitter.process_tag = noop


def _repr(class_to_represent):
    """
    The representation object for all the Header sub-classes. This returns a string in a json-like format which will be ORSO compatible.

    Args:
        class_to_represent (class): The class to be represented.

    Returns:
        (str): A string representation.
    """
    return yaml.dump(class_to_represent, sort_keys=False)


class Header:
    def __repr__(self):
        return _repr(self)


class Orso:
    def __init__(self, creator, data_source, reduction, columns):
        self.creator = creator
        self.data_source = data_source
        self.reduction = reduction
        self.columns = columns

    def __repr__(self):
        return f'# ORSO reflectivity data file | {OSRO_VERSION:.1f} standard | YAML encoding | https://reflectometry.org\n' + _repr(
            self)


class Creator(Header):
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
    def __init__(self, name='Sample'):
        self.name = name


class ValueScalar(Header):
    def __init__(self, magnitude, unit='dimensionless'):
        self.magnitude = magnitude
        if unit == '\xC5':
            unit = 'angstrom'
        self.unit = unit


class ValueRange(Header):
    def __init__(self, min, max, unit='dimensionless'):
        self.min = min
        self.max = max
        if unit == '\xC5':
            unit = 'angstrom'
        self.unit = unit


class Measurement(Header):
    def __init__(self,
                 scheme=None,
                 omega=None,
                 wavelength=None,
                 polarisation=None):
        if scheme is not None:
            self.scheme = scheme
        if omega is not None:
            self.omega = omega
        if wavelength is not None:
            self.wavelength = wavelength
        if polarisation is not None:
            self.polarisation = polarisation


class Experiment(Header):
    def __init__(self, instrument=None, probe=None, sample=None):
        if instrument is not None:
            self.instrument = instrument
        if probe is not None:
            self.probe = probe
        if sample is not None:
            self.sample = sample


class DataSource(Header):
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
    def __init__(self, file):
        self.file = file
        fpath = pathlib.Path(file)
        if not fpath.exists():
            raise FileNotFoundError(f'The file {file} could not be found.')
        else:
            self.creation = datetime.datetime.fromtimestamp(
                fpath.stat().st_ctime).strftime("%Y-%m-%dT%H:%M:%S")


class Files(Header):
    def __init__(self, data_files, reference_files=None):
        self.data_files = data_files
        if reference_files is not None:
            self.reference_files = reference_files


class Reduction(Header):
    def __init__(self, script=None, input_files=None, comment=None):
        self.software = f'ess-{VERSION}'
        if script is not None:
            self.script = script
        if input_files is not None:
            self.input_files = input_files
        if comment is not None:
            self.comment = comment


class Column:
    def __init__(self, quantity, unit='dimensionless', comment=None):
        self.quantity = quantity
        self.unit = unit
        if comment is not None:
            self.comment = comment
