#!/usr/bin/env python3


class BaseDatum:
    """This class stores the features and targets in addition to the ID/name of
    the datum. Note that this class attributes can contain single or batch
    training examples. It is mainly used just to index them."""

    def __init__(self, name, feature, target, extra=None):
        """Initializer.

        Parameters
        ----------
        name : int, list
            The name(s) of the data point for reverse referencing.
        feature : any, list
            Input matrix.
        target : any, list
            Output matrix.
        extra : any, list, optional
            Extra information to be passed to the network.
        """

        self.name = name
        self.feature = feature
        self.target = target
        self.extra = extra
