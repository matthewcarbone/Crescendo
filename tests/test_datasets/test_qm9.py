#!/usr/bin/env python3


from crescendo.datasets.qm9 import QMXDataset


class TestQMXDataset:

    def test_load(self):
        ds = QMXDataset()
        ds.load("data/qm9_test_data")

    def test_loadspectra(self):
        qm8_test = QMXDataset()
        qm8_test.load_qm8_electronic_properties("data/qm8_test_data.txt")
        S1 = qm8_test.qm8_electronic_properties[1]
        assert S1 == [0.43295186, 0.40993872, 0.1832, 0.1832]
