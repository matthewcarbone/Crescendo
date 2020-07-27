#!/usr/bin/env python3


class _Container:
    """A generalized storage unit for featurs, targets, names or metadata.
    Note that each type of container will have it's own compatibility rules
    with different types of data. For example, if the feature and target
    containers are both vector-type, then only a vector-to-vector network
    will be compatible."""

    DATA_TYPE = None
