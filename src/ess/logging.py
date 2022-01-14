# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

import functools
import logging.config
import logging
import inspect
import re
from os import PathLike
from typing import Any, Callable, Literal, MutableMapping, Optional, Union

import scipp as sc
import scippneutron as scn
import structlog


def get_logger(instrument: Optional[str] = None) -> logging.Logger:
    """Return one of ess's loggers.

    :param instrument: If given, return the logger for the specified instrument.
                       Otherwise, return the general ess logger.
    """
    name = 'scipp.ess' + ('.' + instrument if instrument else '')
    return logging.getLogger(name)


_INSTRUMENTS = [
    'amor', 'beer', 'bifrost', 'cspec', 'dream', 'estia', 'freia', 'heimdal', 'loki',
    'magic', 'miracles', 'nmx', 'odin', 'skadi', 'trex', 'v20', 'vespa'
]


def _deduce_instrument_name(f: Any) -> Optional[str]:
    # Assumes package name: ess.<instrument>[.subpackage]
    package = inspect.getmodule(f).__package__
    components = package.split('.', 2)
    try:
        if components[0] == 'ess':
            candidate = components[1]
            if candidate in _INSTRUMENTS:
                return candidate
    except IndexError:
        pass
    return None


def _function_name(f: Callable) -> str:
    if hasattr(f, '__module__'):
        return f'{f.__module__}.{f.__name__}'
    return f.__name__


def log_call(func: Optional[Callable] = None,
             *,
             message: str = None,
             instrument: Optional[str] = None,
             level: int = logging.INFO):
    """Decorator that logs a message every time the function is called.

    Tries to deduce the instrument name from the module of `func`.
    This can be overridden by specifying a name explicitly.
    """

    def deco(f: Callable):
        inst = _deduce_instrument_name(f) if instrument is None else instrument

        @functools.wraps(f)
        def impl(*args, **kwargs):
            if message is not None:
                get_logger(inst).log(level, message)
            else:
                get_logger(inst).log(level, 'Calling %s', _function_name(f))
            return f(*args, **kwargs)

        return impl

    if func is None:
        return deco
    return deco(func)


def set_level_all(logger: logging.Logger, level: Union[str, int]):
    """Sets a log level for a logger and all its handlers."""
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def _make_stream_handler(level: Union[str, int]) -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter('[%(asctime)s] <%(name)s> %(levelname)-8s : %(message)s',
                          datefmt='%Y-%m-%dT%H:%M:%S'))
    return handler


# TODO share handler?
def _make_file_handler(filename: Union[str, PathLike],
                       level: Union[str, int]) -> logging.FileHandler:
    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter('[%(asctime)s] <%(name)s> %(levelname)-8s : %(message)s',
                          datefmt='%Y-%m-%dT%H:%M:%S'))
    return handler


def _configure_root(level, filename, reset):
    root = logging.getLogger()
    if reset:
        for handler in root.handlers:
            root.removeHandler(handler)

    _configure_logger(root, filename, level)


# TODO separate levels for handlers
def _configure_logger(logger: logging.Logger, filename: Optional[Union[str, PathLike]],
                      level: Union[str, int]):
    logger.addHandler(_make_stream_handler(level))
    if filename is not None:
        logger.addHandler(_make_file_handler(filename, level))
    # TODO make it work when not in Jupyter
    logger.addHandler(sc.logging.get_widget_handler())
    logger.setLevel(level)


def _thread_name_abbreviator(logger: logging.Logger, method_name: str,
                             event_dict: MutableMapping[str, Any]):
    name = event_dict['thread_name']
    if name == 'MainThread':
        event_dict['thread_name'] = '0'
    else:
        match = re.match(r'Thread-(\d+)', name)
        if match:
            event_dict['thread_name'] = match[1]
    return event_dict


def _key_value_inserter(logger, method_name, event_dict):
    aux = dict(event_dict)
    del aux['timestamp']
    del aux['level']
    del aux['logger']
    del aux['event']
    rend = structlog.processors.KeyValueRenderer()(logger, method_name, aux)
    if rend:
        event_dict['event'] += ' ' + rend
    return event_dict


def configure(filename: Optional[Union[str, PathLike]] = 'scipp.ess.log',
              level: Union[str, int] = logging.INFO,
              root: Union[bool, Literal['yes', 'no', 'overwrite']] = False):
    """Set up logging for the ess package.

    This function is meant as a helper for application (or notebook) developers.
    It configures the loggers of ess, scippneutron, scipp, and some
    third party packages.
    Calling it from a library can thus mess up a user's setup.

    TODO details
    """
    timestamper = structlog.processors.TimeStamper(fmt='%Y-%m-%dT%H:%M:%S')
    callsite_param_adder = structlog.processors.CallsiteParameterAdder(
        {structlog.processors.CallsiteParameter.THREAD_NAME})
    foreign_pre_chain = [
        timestamper,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.ExtraAdder(),
        callsite_param_adder,
        _thread_name_abbreviator,
    ]
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'incremental': False,  # TODO good choice?
        'formatters': {
            'plain': {
                '()': 'logging.Formatter',
                # TODO thread?
                'format': '[%(asctime)s] <%(name)s> %(levelname)-8s : %(message)s',
                'datefmt': '%Y-%m-%dT%H:%M:%S',
            },
            'json': {
                '()':
                structlog.stdlib.ProcessorFormatter,
                'processors': [
                    # TODO currently prints both logging and structlog fields
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
                'foreign_pre_chain':
                foreign_pre_chain,
            },
        },
        'handlers': {
            'stream': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'plain',
            },
            'json_file': {
                'level': 'INFO',
                'class': 'logging.FileHandler',
                'formatter': 'json',
                'filename': 'test.log.json',
            },
        },
        'loggers': {
            '': {
                'handlers': ['stream', 'json_file'],
                'level': 'INFO',
                'propagate': True,
            }
        }
    })
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            # add timestamp, level, name even though logging handles them separately
            # need them for the JSON formatter
            timestamper,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            callsite_param_adder,
            # _thread_name_abbreviator,
            # structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            _key_value_inserter,
            structlog.stdlib.render_to_log_kwargs,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    return
    if root is True or root in ('yes', 'overwrite'):
        _configure_root(level, filename, reset=root == 'overwrite')
    # TODO don't configure twice -> update scipp
    _configure_logger(sc.get_logger(), filename, level)
    import pooch
    _configure_logger(pooch.get_logger(), filename, level)
    # TODO mantid's own config
    _configure_logger(logging.getLogger('Mantid'), filename, level)


def greet():
    """Log a message showing the versions of important packages."""
    # TODO mantid? what if not used in workflow?
    # TODO add scn.greet()?
    # Import here so we don't import from a partially built package.
    from . import __version__
    get_logger().info(
        '''ess: %s (https://scipp.github.io/ess/)
scippneutron: %s (https://scipp.github.io/scippneutron/)
scipp: %s (https://scipp.github.io/)''', __version__, scn.__version__, sc.__version__)


# TODO time zone in logs
# TODO store file name in datasets?
