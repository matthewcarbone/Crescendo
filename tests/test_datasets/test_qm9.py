#!/usr/bin/env python3

import pytest

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

    def test_pair_qm8_EP(self):
        ds = QMXDataset()
        ds.load("data/qm9_test_data")
        ds.load_qm8_electronic_properties("data/qm8_test_data.txt")
        ds.ml_ready(
            'qm8_EP',
            atom_feature_list=['type', 'hybridization'],
            bond_feature_list=['type']
        )
        ds.ml_data.sort(key=lambda x: x[2])  # sort by ID
        assert ds.ml_data[0][1] == [0.43295186, 0.40993872, 0.1832, 0.1832]

    def test_to_dataloaders(self):
        ds = QMXDataset()
        ds.load("data/qm9_test_data")
        ds.load_qm8_electronic_properties("data/qm8_test_data.txt")
        ds.ml_ready(
            'qm8_EP',
            atom_feature_list=['type', 'hybridization'],
            bond_feature_list=['type']
        )
        ds.get_data_loaders(
            p_tvt=(0.1, 0.1, None), seed=1234, method='random',
            batch_sizes=(32, 32, 32)
        )


@pytest.fixture
def data():
    ds = QMXDataset()
    ds.load("data/qm9_test_data")
    return ds.raw


class TestQM9Datum:

    def test_000001(self, data):
        datum = data[1]
        assert datum.mw == 16.043
        assert datum.smiles == 'C'

    def test_100001(self, data):
        datum = data[100001]
        assert datum.mw == 128.21499999999997
        assert datum.smiles == 'CCC(C)(C)[C@@H](C)C=O'
