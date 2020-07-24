#!/usr/bin/env python3

import numpy as np
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

    def load(self, path):
        """Ingests the .csv file.

        Parameters
        ----------
        path : str
            The path to the .csv file.
        """

        dlog.info(f"Loading data from {path}")
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

        if what not in ['features', 'targets', 'meta', 'id']:
            critical = \
                f"Argument 'data_kind' {what} not valid. Choices " \
                f"are 'features', 'targets', 'meta', 'id'."
            dlog.critical(critical)
            raise RuntimeError(critical)

        loader = CSVLoader(data_kind=what)

        if all(isinstance(xx, int) for xx in columns):
            loader.raw = self.raw.iloc[:, columns]
            return loader

        elif all(isinstance(xx, str) for xx in columns):
            loader.raw = self.raw.loc[:, columns]
            return loader

        critical = \
            "Invalid input for columns. Should be list of str or " \
            "list of int only."
        dlog.critical(critical)
        raise RuntimeError(critical)

    def assert_integrity(
        self, skip_column_name_check=False, skip_row_check=False,
        raise_error=False
    ):
        """Runs integrity checks on the data. Highly recommended in general.
        * Ensures there are no duplicate columns. Note that we require column
          headers. This checks the column names only.
        * Ensures there are no duplicate rows. Expensive numerical comparison.
        Any of these checks can be skipped via specifying the appropriate flag.

        Parameters
        ----------
        skip_column_name_check, skip_row_check : bool
            If True, skips the column name, duplicate row or duplicate column
            checks, respectively. Default is False.
        raise : bool
            If True, will raise a RuntimeError if an integrity check fails.
            Else, will log an error. Default is False
        """

        if not skip_column_name_check:
            condition = np.all(~self.raw.columns.duplicated())
            if not condition:
                error = \
                    "Column name check failed. Check for duplicate columns."
                dlog.error(error)
                if raise_error:
                    raise RuntimeError(error)

        if not skip_row_check:
            initial_shape = self.raw.shape
            new_df = self.raw.drop_duplicates()
            new_shape = new_df.shape
            if new_shape[0] != initial_shape[0]:
                error = \
                    f"Initial number of trials, {initial_shape[0]} != " \
                    f"number of trials after dropping duplciate rows, " \
                    f"{new_shape[0]}"
                dlog.error(error)
                if raise_error:
                    raise RuntimeError(error)



