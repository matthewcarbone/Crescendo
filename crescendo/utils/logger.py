#!/usr/bin/env python3

"""Basic logging module."""

import logging
import sys

logger_string_format = \
    '%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'

formatter = logging.Formatter(logger_string_format)


# https://stackoverflow.com/
# questions/36337244/logging-how-to-set-a-maximum-log-level-for-a-handler
class LevelFilter(logging.Filter):

    def __init__(self, low, high):
        self._low = low
        self._high = high
        logging.Filter.__init__(self)

    def filter(self, record):
        if self._low <= record.levelno <= self._high:
            return True
        return False


def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    handler.addFilter(LevelFilter(10, 20))
    logger.addHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARN)
    handler.setFormatter(formatter)
    handler.addFilter(LevelFilter(30, 50))
    logger.addHandler(handler)

    if log_file is not None:
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.WARN)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger_default = setup_logger('info_stream', log_file=None)
