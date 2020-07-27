#!/usr/bin/env python3

import pandas as pd
import pytest

from crescendo.loaders.base import _CrescendoBaseDataLoader
from crescendo.loaders.text_loaders import CSVLoader
from crescendo.loaders.qm9_loaders import QMXLoader


class TestBaseLoader:

    def test_init(self):
        _ = _CrescendoBaseDataLoader(debug=30)
        _ = _CrescendoBaseDataLoader(debug=-1)
        _ = _CrescendoBaseDataLoader()
        with pytest.raises(Exception):
            _ = _CrescendoBaseDataLoader(debug=1.23)


class TestCSVLoader:

    def test_init(self):
        """Ensures that the testing csv files are loaded properly."""

        df = pd.read_csv("data/csv_test_data/test_10_column.csv")
        features = df.iloc[:, :2]
        targets = df.iloc[:, 2:]
        ds = CSVLoader(features, targets)
        assert ds.features.shape == (1000, 2)
        assert ds.targets.shape == (1000, 8)

    def test_assert_integrity(self):
        """"""

        df = pd.read_csv("data/csv_test_data/test_10_column.csv")
        features = df.iloc[:, :2]
        targets = df.iloc[:, 2:]
        _ = CSVLoader(features, targets)

        df = pd.read_csv("data/csv_test_data/test_10_column.csv")
        features = df.iloc[:, :2]

        # Create dummy features with duplicated columns
        features = pd.concat([features, df.iloc[:, :2]], axis=1)
        targets = df.iloc[:, :6]
        with pytest.raises(Exception):
            _ = CSVLoader(features, targets, raise_error=True)

    def test_check_duplicate_data(self):
        """"""

        df = pd.read_csv("data/csv_test_data/test_10_column.csv")
        features = df.iloc[:, :2]
        features = pd.concat([features, features], axis=0)
        targets = df.iloc[:, 2:]
        targets = pd.concat([targets, targets], axis=0)
        ds = CSVLoader(features, targets, raise_error=True)

        with pytest.raises(Exception):
            ds.check_duplicate_data(on='features')
        with pytest.raises(Exception):
            ds.check_duplicate_data(on='targets')


class TestQMXLoader:

    def test_load(self):
        ds = QMXLoader()
        ds.load("data/qm9_test_data")
