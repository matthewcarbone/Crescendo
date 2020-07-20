#!/usr/bin/env python3

"""Basic logging module."""

import logging

DEFAULT_LOGGER_STRING_FORMAT = '%(asctime)s %(levelname)-8s' \
    '[%(filename)s:%(lineno)d] %(message)s'


def setup_logger(
    name,
    log_file,
    level=logging.INFO,
    logger_string_format=DEFAULT_LOGGER_STRING_FORMAT

):
    """To setup as many loggers as you want:
    https://stackoverflow.com/questions/11232230/
    logging-to-two-files-with-different-settings"""

    if log_file is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(log_file)

    formatter = logging.Formatter(logger_string_format)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


logger_default = setup_logger('info_stream', log_file=None)
