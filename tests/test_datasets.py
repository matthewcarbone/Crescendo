#!/usr/bin/env python3


from crescendo.datasets.qm9 import QMXDataset


class TestQMXDataset:

    def test_load(self):
        ds = QMXDataset()
        ds.load("data/qm9_test_data")
