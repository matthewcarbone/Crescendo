#!/usr/bin/env python3

import pandas as pd
import pytest

from crescendo.datasets.base import _BaseCore, _AllDataset, _SplitDataset
from crescendo.data_containers.standard import ArrayContainer
from crescendo.datasets.text_datasets import CSVDataset
from crescendo.datasets.gm9_dataset import QMXDataset


class TestBaseDatasets:

    def test_init(self):
        _ = _BaseCore()
        _ = _AllDataset()
        _ = _SplitDataset()
        with pytest.raises(Exception):
            _ = _BaseCore(raise_error=1.2)


class TestCSVDatasets:

    def test_init(self):
        """Ensures that the testing csv files are loaded properly."""

        df = pd.read_csv("data/csv_test_data/test_10_column.csv")
        features = df.iloc[:, :2]
        targets = df.iloc[:, 2:]
        ds = CSVDataset(ArrayContainer(features), ArrayContainer(targets))
        assert ds.features.shape == (1000, 2)
        assert ds.targets.shape == (1000, 8)


class TestQMXDataset:

    def test_load(self):
        ds = QMXDataset()
        ds.load("data/qm9_test_data")
