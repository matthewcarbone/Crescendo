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

    def ml_ready(self, features, targets):
        """Converts the raw data into the machine learning-ready format.
        This amounts to peforming the following: 1) selecting the features and
        2) selecting the targets."""

        pass
