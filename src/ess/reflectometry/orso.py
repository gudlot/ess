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


class ORSO:
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
    def __init__(self, name):
        self.name = name


class ValueScalar(Header):
    def __init__(self, magnitude, unit):
        self.magnitude = magnitude
        if unit == '\xC5':
            unit = 'angstrom'
        self.unit = unit


class ValueRange(Header):
    def __init__(self, min, max, unit):
        self.min = min
        self.max = max
        if unit == '\xC5':
            unit = 'angstrom'
        self.unit = unit


class Measurement(Header):
    def __init__(self, scheme, omega, wavelength, polarisation=None):
        self.scheme = scheme
        self.omega = omega
        self.wavelength = wavelength
        if polarisation is not None:
            self.polarisation = polarisation


class Experiment(Header):
    def __init__(self, instrument, probe, sample):
        self.instrument = instrument
        self.probe = probe
        self.sample = sample


class DataSource(Header):
    def __init__(self, owner, facility, experiment_id, experiment_date, title,
                 experiment, measurement):
        self.owner = owner
        self.facility = facility
        self.experiment_id = experiment_id
        self.experiment_date = experiment_date
        self.title = title
        self.experiment = experiment
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
    def __init__(self, script, input_files, comment=None):
        self.software = f'ess-{VERSION}'
        self.comment = ''
        if script is None:
            self.comment += 'No script is defined, a Jupyter notebook was used for reduction.'
        else:
            self.script = script
        self.input_files = input_files
        if comment is not None:
            self.comment += comment


class Column:
    def __init__(self, quantity, unit, comment=None):
        self.quantity = quantity
        self.unit = unit
        if comment is not None:
            self.comment = comment
