#!/usr/bin/env python3

from crescendo.loaders.base import BaseDataset
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
