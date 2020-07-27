#!/usr/bin/env python3

import pandas as pd
import pytest

from crescendo.loaders.base import _BaseCore, _AllDataset, _SplitDataset
from crescendo.data_containers.standard import ArrayContainer
from crescendo.loaders.text_loaders import CSVLoader
from crescendo.loaders.qm9_loaders import QMXLoader


class TestBaseLoaders:

    def test_init(self):
        _ = _BaseCore()
        _ = _AllDataset()
        _ = _SplitDataset()
        with pytest.raises(Exception):
            _ = _BaseCore(raise_error=1.2)


class TestCSVLoader:

    def test_init(self):
        """Ensures that the testing csv files are loaded properly."""

        df = pd.read_csv("data/csv_test_data/test_10_column.csv")
        features = df.iloc[:, :2]
        targets = df.iloc[:, 2:]
        ds = CSVLoader(ArrayContainer(features), ArrayContainer(targets))
        assert ds.features.shape == (1000, 2)
        assert ds.targets.shape == (1000, 8)


class TestQMXLoader:

    def test_load(self):
        ds = QMXLoader()
        ds.load("data/qm9_test_data")
