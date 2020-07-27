#!/usr/bin/env python3

from crescendo.utils.logger import logger_default as dlog


class _CrescendoBaseDataLoader:

    def __init__(self, debug=-1, raise_error=False):
        """Base initializer.

        Parameters
        ----------
        debug : int
            If set to -1 (default) then we're not in debug mode. If some
            integer > 0, then that is the number of total data points loaded
            into self.raw.
        raise_error : bool
            If true, will raise errors and terminate the program on non
            critical errors
        """

        self.raw = None
        self.debug = debug
        self.raise_error = raise_error

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, d):
        assert isinstance(d, int)
        assert d == -1 or d > 0
        dlog.info(f"Loader debug variable set to {d}")
        self._debug = d

    @property
    def raise_error(self):
        return self._raise_error

    @raise_error.setter
    def raise_error(self, r):
        assert isinstance(r, bool)
        self._raise_error = r

    def load(self):
        """Optional, depending on the loader type."""

        raise NotImplementedError

    def init_raw(self):
        """Initializes the raw attribute from the features and targets. Takes
        optional arguments depending on the data type, possibly on how to
        construct the features and targets themselves."""

        raise NotImplementedError
