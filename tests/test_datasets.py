#!/usr/bin/env python3

import numpy as np
import pytest

from crescendo.loaders.base import _CrescendoBaseDataLoader
from crescendo.loaders.text_loaders import CSVLoader


class TestBaseLoader:

    def test_base_init(self):
        _ = _CrescendoBaseDataLoader()


class TestCSVLoader:

    def test_csv_load(self):
        """Ensures that the testing csv files are loaded properly."""

        ds = CSVLoader()
        ds.load("data/test_3_column.csv", data_kind='all')
        assert ds.raw.shape[0] == 1000
        assert ds.raw.shape[1] == 3

        ds = CSVLoader()
        ds.load("data/test_10_column.csv", data_kind='features')
        assert ds.raw.shape[0] == 1000
        assert ds.raw.shape[1] == 10

        ds = CSVLoader()
        with pytest.raises(Exception):
            ds.load("data/test_10_column.csv", data_kind='incorrect_data_kind')

    def test_get(self):
        """Tests the get method."""

        ds = CSVLoader()
        ds.load("data/test_10_column.csv", data_kind='features')
        features = ds.get('features', [0, 1, 2])
        assert features.raw.shape[0] == 1000
        assert features.raw.shape[1] == 3

        ds = CSVLoader()
        ds.load("data/test_10_column.csv", data_kind='features')
        _ = ds.get('features', [0, 1, 2])
        _ = ds.get('targets', [0, 1, 2])
        _ = ds.get('meta', [0, 1, 2])
        with pytest.raises(Exception):
            _ = ds.get('incorrect_data_kind', [0, 1, 2])

    def test_assert_integrity(self):
        """"""

        ds = CSVLoader()
        ds.load("data/test_10_column.csv", data_kind='features')
        ds.assert_integrity(raise_error=True)
        features = ds.get(
            'features', ['feature_1', 'feature_2', 'feature_7', 'feature_9']
        )
        features.assert_integrity(raise_error=True)

        targets = ds.get(
            'targets', [1, 2, 3, 5, 4, 3]
        )

        with pytest.raises(Exception):
            targets.assert_integrity(raise_error=True)
