"""BSD 3-Clause License

Copyright (c) 2023, Matthew R. Carbone, Stepan Fomichev & John Sous
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from contextlib import contextmanager
import sys
from warnings import warn

from loguru import logger


NO_DEBUG_LEVELS = ["INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]


def generic_filter(names):
    if names == "all":
        return None

    def f(record):
        return record["level"].name in names

    return f


# DEBUG_FMT_WITH_MPI_RANK = (
#     "<fg #808080>{time:YYYY-MM-DD HH:mm:ss.SSS} "
#     "{name}:{function}:{line}</> "
#     "|<lvl>{level: <10}</>|%s<lvl>{message}</>" % RANK
# )


# DEBUG_FMT_WITHOUT_MPI_RANK = (
#     "<fg #808080>{time:YYYY-MM-DD HH:mm:ss.SSS} "
#     "{name}:{function}:{line}</> "
#     "|<lvl>{level: <10}</>| <lvl>{message}</>"
# )


# DEBUG_FMT_WITHOUT_MPI_RANK = (
#     "<fg #808080>{time:YYYY-MM-DD HH:mm:ss}</> <lvl>{message}</>"
# )

# INFO_FMT_WITHOUT_MPI_RANK = (

# )

# WARN_FMT_WITHOUT_MPI_RANK = (
#     "<fg #808080>{time:YYYY-MM-DD HH:mm:ss.SSS} "
#     "{name}:{function}:{line}</> "
#     "[<lvl>{level: <10}</>] <lvl>{message}</>"
# )

format_mapping = {
    "DEBUG": "[<lvl>D</>] <lvl>{message}</>",
    "INFO": "[<lvl>I</>] <lvl>{message}</>",
    "SUCCESS": "[<lvl>S</>] <lvl>{message}</>",
    "WARNING": "[<lvl>W</>] <lvl>{message}</>",
    "ERROR": "[<lvl>E</>] <lvl>{message}</>",
    "CRITICAL": "[<lvl>C</>] <lvl>{message}</>",
}


def configure_loggers(
    levels=["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
    enable_python_standard_warnings=False,
):
    """Configures the ``loguru`` loggers. Note that the loggers are initialized
    using the default values by default.

    .. important::

        ``logger.critical`` `always` terminates the program, either through
        ``COMM.MPI_Abort()`` if ``run_as_mpi`` is True, or ``sys.exit(1)``
        otherwise.

    Parameters
    ----------
    levels : list, optional
        Description
    enable_python_standard_warnings : bool, optional
        Raises dummy warnings on ``logger.warning`` and ``logger.error``.
    """

    logger.remove(None)  # Remove ALL handlers

    for level in levels:
        logger.add(
            sys.stdout
            if level in ["DEBUG", "INFO", "SUCCESS", "WARNING"]
            else sys.stderr,
            colorize=True,
            filter=generic_filter([level]),
            format=format_mapping[level],
        )

    if enable_python_standard_warnings:
        logger.add(lambda _: warn("DUMMY WARNING"), level="WARNING")
        logger.add(lambda _: warn("DUMMY ERROR"), level="ERROR")


def DEBUG():
    """Quick helper to enable DEBUG mode."""

    configure_loggers()


def _TESTING_MODE():
    """Enables a testing mode where loggers are configured as usual but where
    the logger.warning and logger.error calls actually also raise a dummy
    warning with the text "DUMMY WARNING" and "DUMMY ERROR", respectively.
    Used for unit tests."""

    configure_loggers(enable_python_standard_warnings=True)


def DISABLE_DEBUG():
    """Quick helper to disable DEBUG mode."""

    configure_loggers(levels=NO_DEBUG_LEVELS)


@contextmanager
def disable_logger():
    """Context manager for disabling the logger."""

    logger.disable("")
    try:
        yield None
    finally:
        logger.enable("")


@contextmanager
def _testing_mode():
    _TESTING_MODE()
    try:
        yield None
    finally:
        DEBUG()


@contextmanager
def debug():
    DEBUG()
    try:
        yield None
    finally:
        DISABLE_DEBUG()


DISABLE_DEBUG()
