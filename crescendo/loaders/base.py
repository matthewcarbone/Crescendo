#!/usr/bin/env python3


class _CrescendoBaseDataLoader:

    def __init__(self, *, debug=False):
        """Base initializer.

        Parameters
        ----------
        debug : int
            If set to -1 (default) then we're not in debug mode. If some
            integer > 0, then that is the number of total data points loaded
            in to self.raw.
        """

        self.path = None
        self.raw = None
        self.data_kind = None
        self.debug = debug

    def load(self):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def assert_integrity(self):
        raise NotImplementedError
