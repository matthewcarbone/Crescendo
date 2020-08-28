#!/usr/bin/env python3

import copy
import pytest
import shutil

from crescendo.datasets.qm9 import QM9Dataset
from crescendo.defaults import QM9_TEST_DATA_PATH, QM8_TEST_DATA_PATH


@pytest.fixture
def ds():
    ds = QM9Dataset()
    ds.load(QM9_TEST_DATA_PATH)
    return ds


class TestQMXDataset:

    def test_load(self, ds):
        _ = ds

    def test_loadspectra(self, ds):
        ds.load_qm8_electronic_properties(
            QM8_TEST_DATA_PATH, selected_properties=[0, 13, 14, 15, 16]
        )
        assert ds[1].qm8properties == [0.43295186, 0.40993872, 0.1832, 0.1832]

    def test_save_load_state(self):
        ds = QM9Dataset(dsname="TESTDS")
        ds.load(QM9_TEST_DATA_PATH)
        ds_COPY = copy.deepcopy(ds)
        ds.save_state(directory="TEST_DIR", override=False)
        new_ds = QM9Dataset()
        new_ds.load_state("TESTDS", directory="TEST_DIR")
        shutil.rmtree('TEST_DIR')

        for key, value in new_ds.__dict__.items():
            if key == 'raw':
                for qm9ID, datum in ds_COPY.raw.items():
                    assert datum.smiles == new_ds[qm9ID].smiles
            else:
                assert getattr(ds_COPY, key) == value


class TestQM9Datum:

    def test_000001(self, ds):
        datum = ds.raw[1]
        assert datum.smiles[0] == 'C'
        assert datum.smiles[1] == 'C'

    def test_100001(self, ds):
        datum = ds.raw[100001]
        assert datum.smiles[0] == 'CCC(C)(C)C(C)C=O'
        assert datum.smiles[1] == 'CCC(C)(C)[C@@H](C)C=O'
