#!/usr/bin/env python3

import numpy as np
import pandas as pd

from crescendo.loaders.base import _CrescendoBaseDataLoader
from crescendo.utils.logger import logger_default as dlog


class CSVLoader(_CrescendoBaseDataLoader):
    """A simple dataset designed to initialize a ML database from csv
    features and targets."""

    def __init__(self, features, targets, **kwargs):
        """Initializes the CSVLoader object with pd.DataFrame features and
        targets that the user loads beforehand.

        Parameters
        ----------
        features, targets : pd.DataFrame
            Note that the features and targets may contain different numbers
            of columns, but must have the same number of rows (data points).
            Each column corresponds to a different property of a single data
            point which may be a feature or target.
        """

        super().__init__(**kwargs)
        self.features = features
        self.targets = targets

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, f):

        if not isinstance(f, pd.DataFrame):
            critical = \
                f"Features are of type {type(f)} but must be pd.DataFrame"
            dlog.critical(critical)
            raise RuntimeError(critical)

        condition = np.all(~f.columns.duplicated())
        if not condition:
            error = "Duplicate columns found in features"
            dlog.error(error)
            if self.raise_error:
                raise RuntimeError(error)

        s1 = f.shape[0]
        s2 = f.shape[1]
        dlog.info(f"Init features of shape {s1} x {s2}")
        self._features = f

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, t):

        if not isinstance(t, pd.DataFrame):
            critical = \
                f"Targets are of type {type(t)} but must be pd.DataFrame"
            dlog.critical(critical)
            raise RuntimeError(critical)

        condition = np.all(~t.columns.duplicated())
        if not condition:
            error = "Duplicate columns found in features"
            dlog.error(error)
            if self.raise_error:
                raise RuntimeError(error)

        s1 = t.shape[0]
        s2 = t.shape[1]
        dlog.info(f"Init targets of shape {s1} x {s2}")
        self._targets = t

    def check_duplicate_data(self, on):
        """Runs integrity checks on the data. Highly recommended in general.
        Ensures there are no duplicate data points.

        Parameters
        ----------
        on : {'features', 'targets'}
            Which set of data to run the integrity checks on. Both the features
            and targets are pd.DataFrame objects, but it is possible that
            the user is confident in one and not the other, or that one of the
            DataFrames is extremely large, and thus it will be hard to run on
            the larger one. Thus, while asserting the integrity of both the
            features and targets is recommended, the user can choose which
            if not both.
        """

        if on == 'features':
            raw = self.features
        elif on == 'targets':
            raw = self.targets
        else:
            critical = \
                "Can only run integrity checks on 'features' or 'targets' " \
                f"but {on} was specified"
            dlog.critical(critical)
            raise RuntimeError(critical)

        initial_shape = raw.shape
        new_df = raw.drop_duplicates()  # Returns copy by default
        new_shape = new_df.shape
        if new_shape[0] != initial_shape[0]:
            error = \
                f"Initial number of trials, {initial_shape[0]} != " \
                "number of trials after dropping duplciate rows, " \
                f"{new_shape[0]}"
            dlog.error(error)
            if self.raise_error:
                raise RuntimeError(error)
