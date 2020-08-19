#!/usr/bin/env python3

import os
from crescendo.utils.logger import logger_default as dlog


def flatten_list(full_list):
    """Flattens a list of list into a single list using list comprehension.

    Parameters
    ----------
    full_list : list
        A list of lists.

    Returns
    -------
    list
        The flattened list.
    """

    return [item for sublist in full_list for item in sublist]


def intersection(list1, list2):
    """Returns the intersection of two lists.

    Parameters
    ----------
    list1, list2 : list

    Returns
    -------
    list
    """

    return list(set(list1) & set(list2))


def check_for_environment_variable(var):
    """Checks the os.environ dictionary for the specified environment
    variable. If it exists, returns the path, else raises a ValueError and
    logs a critical level error to the logger."""

    qm9_directory = os.environ.get(var, None)
    if qm9_directory is None:
        error_msg = \
            f"Environment variable {var} not found and no path specified"
        dlog.critical(error_msg)
        raise RuntimeError(error_msg)
    return qm9_directory
