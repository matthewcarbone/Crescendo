#!/usr/bin/env python3

from crescendo.utils.logger import logger_default as dlog


class _CrescendoBaseDataLoader:

    def __init__(self, *, debug=False, data_kind='unknown'):
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

        self.path = None
        self.raw = None
        self.debug = debug

        dlog.info(f"Initializng Loader for {data_kind} data")
        if data_kind not in ['features', 'targets', 'all', 'meta', 'id']:
            critical = \
                f"Argument 'data_kind' {data_kind} not valid. Choices " \
                f"are 'features', 'targets', 'all', 'meta', 'id'."
            dlog.critical(critical)
            raise RuntimeError(critical)
        self.data_kind = data_kind

    def load(self):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def assert_integrity(self):
        raise NotImplementedError
