#!/usr/bin/env python3

import numpy as np
import pandas as pd

from crescendo.data_containers.base import _Container
from crescendo.utils.logger import logger_default as dlog


class ArrayContainer(_Container, pd.DataFrame):

    DATA_TYPE = 'VECTOR'

    def __init__(self, *args, **kwargs):
        """Ingests a pandas DataFrame-compatible object."""

        super().__init__(*args, **kwargs)

        s1 = self.shape[0]
        s2 = self.shape[1]
        dlog.info(f"Init PandasContainer of shape {s1} x {s2}")

    def assert_integrity_duplicate_rows_numerical(self):
        """Asserts there are no duplicate rows in the ingested dataframe."""

        initial_shape = self.shape
        new_df = self.drop_duplicates()  # Returns copy by default
        new_shape = new_df.shape
        if new_shape[0] != initial_shape[0]:
            critical = \
                f"Initial number of trials, {initial_shape[0]} != " \
                "number of trials after dropping duplciate rows, " \
                f"{new_shape[0]}"
            dlog.critical(critical)
            raise RuntimeError(critical)

    def assert_integrity_duplicate_columns(self):
        """Asserts there are no duplicate rows in the ingested dataframe."""

        condition = np.all(~self.columns.duplicated())
        if not condition:
            critical = "Duplicate columns found in data"
            dlog.critical(critical)
            raise RuntimeError(critical)
