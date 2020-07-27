#!/usr/bin/env python3

import pandas as pd
import pytest

from crescendo.data_containers.base import _Container
from crescendo.data_containers.standard import ArrayContainer


class TestBaseContainer:

    def test_init(self):
        c = _Container()
        assert c.DATA_TYPE is None


class TestArrayContainer:

    def test_init(self):
        """Ensures that the testing csv files are loaded properly."""

        df = pd.read_csv("data/csv_test_data/test_10_column.csv")
        features = df.iloc[:, :2]
        _ = ArrayContainer(features)

    def test_integrity(self):
        """Ensures that the testing csv files are loaded properly."""

        df = pd.read_csv("data/csv_test_data/test_10_column.csv")
        features = df.iloc[:, :2]
        targets = df.iloc[:, 2:]

        f = ArrayContainer(features)
        f.assert_integrity_duplicate_rows_numerical()
        f.assert_integrity_duplicate_columns()

        t = ArrayContainer(targets)
        t.assert_integrity_duplicate_rows_numerical()
        t.assert_integrity_duplicate_columns()

        bad_features = pd.concat([features, features], axis=1)
        t = ArrayContainer(bad_features)
        t.assert_integrity_duplicate_rows_numerical()
        with pytest.raises(Exception):
            t.assert_integrity_duplicate_columns()

        bad_features2 = pd.concat([features, features], axis=0)
        t = ArrayContainer(bad_features2)
        t.assert_integrity_duplicate_columns()
        with pytest.raises(Exception):
            t.assert_integrity_duplicate_rows_numerical()


"""
class TestQMXLoader:

    def test_load(self):
        ds = QMXLoader()
        ds.load("data/qm9_test_data")
"""
