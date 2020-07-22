#!/usr/bin/env python3

import pandas as pd

from crescendo.loaders.base import BaseDataset
from crescendo.utils.logger import logger_default as dlog


class CSVDataSet(BaseDataset):
    """A simple loader designed to read data from a .csv file using pandas.
    The general approach is that each row of the .csv is a data point and
    each column is a feature. The user has to specify which columns to use as
    features and which columns to use as targets. The entire .csv is initially
    ingested and then only the relevant columns are selected."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, path, header=True, usecols=None):
        """Ingests the .csv file.

        Parameters
        ----------
        path : str
            The path to the .csv file.
        header : bool
            Whether or not the raw text data has headers for the columns.
            This argument is passed directly to pd.read_csv. Default is True.
        usecols : list, optional
            A list of integers indicating the columns to load. Note that this
            does not select any feature or target, it will simply downsample
            the entire dataset by using less columns. This argument is passed
            directly to pd.read_csv. Default is None.
        """

        dlog.info(f"Loading data from {path}")
        dlog.info(f"Use headers is {header}")

        if usecols is not None:
            if not all(isinstance(xx, int) for xx in usecols):
                dlog.error(
                    "Argument usecols is not all of type int, expect unknown "
                    "behavior"
                )

        self.raw = pd.read_csv(path, header=header, usecols=usecols)

        s0 = self.raw.shape[0]
        s1 = self.raw.shape[1]
        dlog.info(f"Read DataFrame is of shape {s0} (rows) x {s1} (columns)")

    def _select_specific_columns(self, cols, datatype='Unknown'):
        """Helper method that works generally for any subset of columns.
        Selects the columns based on the logic in ml_ready."""

        if datatype == 'Unknown':
            dlog.warning("Datatype not specified, logging as 'Unknown'")

        if all(isinstance(xx, int) for xx in cols):
            dlog.info(
                f"{datatype} list is of type integer, selecting using iloc"
            )
            return self.raw.iloc[:, cols]

        if all(isinstance(xx, str) for xx in cols):
            dlog.info(
                f"{datatype} list is of type string, selecting using loc"
            )
            return self.raw.loc[:, cols]

        if len(cols) == 2:
            if cols[0] is None and isinstance(cols[1], int):
                dlog.info(
                    f"{datatype} list will select via .iloc[:, :{cols[1]}]"
                )
                if cols[1] < 0:
                    dlog.warning(
                        f"Note {cols[1]} < 0, ensure this behavior is desired"
                    )
                return self.raw.iloc[:, :cols[1]]
            elif isinstance(cols[0]) and cols[1] is None:
                dlog.info(
                    f"{datatype} list will select via .iloc[:, {cols[0]}:]"
                )
                if cols[0] < 0:
                    dlog.warning(
                        f"Note {cols[0]} < 0, ensure this behavior is desired"
                    )
                return self.raw.iloc[:, cols[0]:]

        critical = \
            "Unknown cols input - should be list of str, list of int, or of " \
            "the form [None, int] or [int, None] for list slicing. See " \
            "'ml_ready' docstring for more details."
        dlog.critical(critical)
        raise NotImplementedError(critical)

    def _select_features_targets(self, features, targets):
        """Helper method for parsing the input to ml_ready."""

        # Ensures that the iputs are of type list
        if not isinstance(features, list) or not isinstance(targets, list):
            critical = \
                f"Selected features/targets are not of type list: " \
                f"({type(features)}/{type(targets)})"
            dlog.critical(critical)
            raise RuntimeError(critical)

        f_data = self._select_specific_columns(features, datatype='Features')
        t_data = self._select_specific_columns(targets, datatype='Targets')
        dlog.info(
            f"Feature and target DataFrames are of shapes "
            f"{f_data.shape[0]}/{f_data.shape[1]} and "
            f"{t_data.shape[0]}/{t_data.shape[0]}"
        )
        return f_data, t_data

    def ml_ready(self, features, targets):
        """Converts the raw data into the machine learning-ready format.
        This amounts to peforming the following: 1) selecting the features,
        2) selecting the targets, and 3) converting the data into a format
        that can be processed by pytorch.

        features, targets : list
            If a list of strings, uses the loc method to select the columns
            by header name. If a list of integers, uses the iloc method to
            select columns by index. A third option is feeding the input a
            term like [5, None], which will translate to .iloc[:, 5:] in
            list comprehension. Similarly, [None, 3] translates to
            .iloc[:, :3].
        """

        # The returned f_data and t_data are the feature and target DataFrames
        f_data, t_data = self._select_features_targets(features, targets)
