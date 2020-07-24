#!/usr/bin/env python3

from crescendo.utils.logger import logger_default as dlog


class _CrescendoBaseDataLoader:

    def __init__(self, *, debug=-1, data_kind='unknown'):
        """Base initializer.

        Parameters
        ----------
        debug : int
            If set to -1 (default) then we're not in debug mode. If some
            integer > 0, then that is the number of total data points loaded
            in to self.raw.
        data_kind : {'features', 'targets', 'all', 'meta'}
            The type of data being loaded. This is critical as the label will
            be used downstream for model compatibility.
            * 'features': the loaded csv corresponds to the features of the
                data.
            * 'targets': the loaded csv corresponds to the targets of the data.
            * 'all': the loaded csv contains all data (features and targets
                in different columns).
            * 'meta': the loaded csv contains extra data that is neither the
                features nor targets.
            * 'id': an optional "column" that identifies the trial by some
                label other than the index (row).
        """

        self.raw = None
        self.debug = debug
        self.data_kind = data_kind

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
    def data_kind(self):
        return self._data_kind

    @data_kind.setter
    def data_kind(self, data_kind):
        assert isinstance(data_kind, str)
        if data_kind not in ['features', 'targets', 'all', 'meta', 'id']:
            critical = \
                f"Argument 'data_kind' {data_kind} not valid. Choices " \
                f"are 'features', 'targets', 'all', 'meta', 'id'."
            dlog.critical(critical)
            raise RuntimeError(critical)
        dlog.info(f"Initializing Loader for {data_kind} data")
        self._data_kind = data_kind

    def load(self):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def assert_integrity(self):
        raise NotImplementedError
