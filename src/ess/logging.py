# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

import functools
import logging.config
import logging
import inspect
from os import PathLike
from typing import Any, Callable, List, Literal, Optional, Union

import scipp as sc
import scippneutron as scn


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


def _make_file_handler(filename: Union[str, PathLike],
                       level: Union[str, int]) -> logging.FileHandler:
    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter('[%(asctime)s] <%(name)s> %(levelname)-8s : %(message)s',
                          datefmt='%Y-%m-%dT%H:%M:%S'))
    return handler


def _make_widget_handler(level: Union[str, int]) -> sc.logging.WidgetHandler:
    return sc.logging.WidgetHandler(level=level, widget=sc.logging.LogWidget())


def _configure_root(handlers: List[logging.Handler], level: Union[str, int], reset):
    root = logging.getLogger()
    if reset:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for filt in list(root.filters):
            root.removeFilter(filt)

    _configure_logger(root, handlers, level)


def _configure_logger(logger: logging.Logger, handlers: List[logging.Handler],
                      level: Union[str, int]):
    for handler in handlers:
        logger.addHandler(handler)
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

def _make_handlers(filename: Optional[Union[str, PathLike]],
                   file_level: Union[str, int], stream_level: Union[str, int],
                   widget_level: Union[str, int]) -> List[logging.Handler]:
    handlers = [_make_stream_handler(stream_level)]
    if filename is not None:
        handlers.append(_make_file_handler(filename, file_level))
    if sc.utils.running_in_jupyter():
        handlers.append(_make_widget_handler(widget_level))
    return handlers


def _base_level(levels: List[Union[str, int]]) -> int:
    return min((logging.getLevelName(level) if isinstance(level, str) else level
                for level in levels))


def configure(filename: Optional[Union[str, PathLike]] = 'scipp.ess.log',
              file_level: Union[str, int] = logging.INFO,
              stream_level: Union[str, int] = logging.WARNING,
              widget_level: Union[str, int] = logging.INFO,
              root: Union[bool, Literal['yes', 'no', 'overwrite']] = False):
    """Set up logging for the ess package.

    This function is meant as a helper for application (or notebook) developers.
    It configures the loggers of ess, scippneutron, scipp, and some
    third party packages.
    Calling it from a library can thus mess up a user's setup.

    TODO details
    """
    handlers = _make_handlers(filename, file_level, stream_level, widget_level)
    base_level = _base_level([file_level, stream_level, widget_level])
    if root is True or root in ('yes', 'overwrite'):
        _configure_root(handlers, base_level, reset=root == 'overwrite')
    # TODO don't configure twice -> update scipp
    import pooch
    for logger in (sc.get_logger(), pooch.get_logger(), logging.getLogger('Mantid')):
        _configure_logger(logger, handlers, base_level)
    # TODO mantid's own config


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
