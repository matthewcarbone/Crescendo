#!/usr/bin/env python3

import numpy as np
import pytest

from crescendo.datasets.base import BaseDataset
from crescendo.loaders.text_loaders import CSVDataSet


class TestDataset:

    def test_base_init(self):
        ds = BaseDataset(name='my_test')
        assert ds.raw is None
        assert ds.ml_data is None
        assert ds.name == 'my_test'

    def test_csv_load(self):
        """Ensures that the testing csv files are loaded properly."""

        ds = CSVDataSet()
        ds.load("data/test_3_column.csv")
        assert ds.raw.shape[0] == 1000
        assert ds.raw.shape[1] == 3

        ds = CSVDataSet()
        ds.load("data/test_10_column.csv")
        assert ds.raw.shape[0] == 1000
        assert ds.raw.shape[1] == 10

    def test__select_specific_columns_errors(self):
        """Ensures proper use case for the helper method
        _select_specific_columns."""

        ds = CSVDataSet()
        ds.load("data/test_10_column.csv")

        with pytest.raises(Exception):
            ds._select_specific_columns(cols=["1", 1, 2, 3])

        with pytest.raises(Exception):
            ds._select_specific_columns(cols=[1.0, 1, 2, 3])

        with pytest.raises(Exception):
            ds._select_specific_columns(cols=[np.array([1, 2, 3]), 1, 2, 3])

        with pytest.raises(Exception):
            ds._select_specific_columns(cols=["1", "2", "3", "4", 5])

    def test__select_specific_columns_selection(self):
        """Makes sure the user can select columns in the appropriate ways."""

        ds = CSVDataSet()
        ds.load("data/test_10_column.csv")
        ds.ml_ready(
            [0, 1, 2], ["feature_3", "feature_4", "feature_5"],
            assert_integrity=True, assert_level='raise'
        )

        ds = CSVDataSet()
        ds.load("data/test_10_column.csv")
        with pytest.raises(Exception):
            ds.ml_ready(
                [0, 1, 2], [None, 5], assert_integrity=True,
                assert_level='raise'
            )
