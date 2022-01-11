# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

import functools
import logging
import inspect
from typing import Any, Callable, Optional

import scipp as sc
import scippneutron as scn


def get_logger(instrument: Optional[str] = None) -> logging.Logger:
    """Return one of ess's loggers.

    :param instrument: If given, return the logger for the specified instrument.
                       Otherwise, return the general ess logger.
    """
    name = 'scipp.ess' + ('.' + instrument if instrument else '')
    return logging.getLogger(name)


_INSTRUMENTS = ['amor', 'beer', 'bifrost', 'cspec', 'dream', 'estia', 'freia', 'heimdal', 'loki',
                'magic', 'miracles', 'nmx', 'odin', 'skadi', 'trex', 'v20', 'vespa']


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


def log_call(func: Optional[Callable] = None, *, instrument: Optional[str] = None,
             level: int = logging.INFO):
    """Decorator that logs a message every time the function is called.

    Tries to deduce the instrument name from the module of `func`.
    This can be overridden by specifying a name explicitly.
    """

    def deco(f: Callable):
        inst = _deduce_instrument_name(f) if instrument is None else instrument

        @functools.wraps(f)
        def impl(*args, **kwargs):
            get_logger(inst).log(level, 'Calling %s', _function_name(f))
            return f(*args, **kwargs)

        return impl

    if func is None:
        return deco
    return deco(func)


def set_level_all(logger: logging.Logger, level: int):
    """Sets a log level for a logger and all its handlers."""
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def _handle_same_as_scipp(logger: logging.Logger):
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    for handler in sc.get_logger().handlers:
        logger.addHandler(handler)


def _configure_3rd_party(logger: logging.Logger, level: int):
    _handle_same_as_scipp(logger)
    set_level_all(logger, level)


def configure(level: int = logging.INFO):
    """Set up logging for the ess package.

    This function is meant as a helper for application (or notebook) developers.
    It configures the loggers of ess, scippneutron, scipp, and some third party packages.
    Calling it from a library can thus mess up a user's setup.
    """
    import pooch

    set_level_all(sc.get_logger(), level)

    _configure_3rd_party(pooch.get_logger(), level)
    # TODO mantid's own config
    _configure_3rd_party(logging.getLogger('Mantid'), level)

    # TODO file, console (stdlog or stderr?)


def greet():
    """Log a message showing the versions of important packages."""
    # TODO mantid? what if not used in workflow?
    # TODO add scn.greet()?
    from . import __version__
    get_logger().info('ESS v%s\nscippneutron v%s\nscipp v%s',
                      __version__, scn.__version__, sc.__version__)
