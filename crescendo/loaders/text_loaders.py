#!/usr/bin/env python3

import pandas as pd

from crescendo.loaders.base import _CrescendoBaseDataLoader
from crescendo.utils.logger import logger_default as dlog


class CSVLoader(_CrescendoBaseDataLoader):
    """A simple loader designed to read data from a .csv file using pandas.
    The general approach is that each row of the .csv is a data point and
    each column is a feature. Each instance of the CSVLoader should contain
    either the features or targets for the problem. Alternatively, the user
    can load in all data (featuers and targets) and then specify how to split
    that data into features and targets, returning two instances of
    CSVLoader."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, path, data_kind):
        """Ingests the .csv file.

        Parameters
        ----------
        path : str
            The path to the .csv file.
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
        """

        if data_kind not in ['features', 'targets', 'all', 'meta']:
            critical = \
                f"Argument 'data_kind' {data_kind} not valid. Choices " \
                f"are 'features', 'targets', 'all', 'meta'."
            dlog.critical(critical)
            raise RuntimeError(critical)
        self.data_kind = data_kind

        dlog.info(f"Loading data [{data_kind}] from {path}")
        self.raw = pd.read_csv(path)
        s0 = self.raw.shape[0]
        s1 = self.raw.shape[1]
        dlog.info(f"Read DataFrame is of shape {s0} (rows) x {s1} (columns)")

    def get(self, what, columns):
        """If the loaded data corresponds to data_type 'all', this method
        returns a subset of the columns as a new instance of CSVLoader.

        Parameters
        ----------
        what : {'features', 'targets', 'meta'}
            Indicates the data_type attribute of the returned CSVLoader class.
        columns : list
            Either a list of strings or list of integers corresponding to the
            columns of the returned CSVLoader.raw. Note that if the list is
            all integers, the data is accessed via self.raw.iloc and if
            the list is all strings, the data is accessed via self.raw.loc.

        Returns
        -------
        CSVLoader
            A fresh instance of the CSVLoader with the data_kind attribute set
            via the 'what' argument.
        """

        if self.data_kind != 'all':
            dlog.warning(
                f"Sampling {what} from a dataset labeled {self.data_kind}, "
                f"the user should ensure this is intended."
            )

        if what not in ['features', 'targets', 'meta']:
            critical = \
                f"Argument 'data_kind' {what} not valid. Choices " \
                f"are 'features', 'targets', 'meta'."
            dlog.critical(critical)
            raise RuntimeError(critical)

        loader = CSVLoader()
        loader.data_kind = what

        if all(isinstance(xx, int) for xx in columns):
            loader.raw = self.raw.loc[:, columns]
            return loader

        elif all(isinstance(xx, str) for xx in columns):
            loader.raw = self.raw.iloc[:, columns]
            return loader

        critical = \
            "Invalid input for columns. Should be list of str or " \
            "list of int only."
        dlog.critical(critical)
        raise RuntimeError(critical)
