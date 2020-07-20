#!/usr/bin/env python3


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
