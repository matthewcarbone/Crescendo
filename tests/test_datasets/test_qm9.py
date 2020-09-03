#!/usr/bin/env python3

import copy
import numpy as np
import pytest
import shutil

from crescendo.datasets.qm9 import QM9Dataset, QM9GraphDataset
from crescendo.defaults import QM9_TEST_DATA_PATH, QM8_TEST_DATA_PATH


@pytest.fixture
def ds():
    ds = QM9Dataset()
    ds.load(QM9_TEST_DATA_PATH)
    return ds


class TestQM9Dataset:

    def test_load(self, ds):
        _ = ds

    def test_load_qm8_properties(self, ds):
        ds.load_qm8_electronic_properties(QM8_TEST_DATA_PATH)
        assert ds[1].qm8properties == [
            0.43295186, 0.43295958, 0.24972825, 0.24973648, 0.43021753,
            0.43023558, 0.18143600, 0.18150153, 0.40985825, 0.40988403,
            0.17772250, 0.17741930, 0.40993137, 0.40993872, 0.18320000,
            0.18320000
        ]

    def test_save_load_state(self):
        ds = QM9Dataset(dsname="TESTDS")
        ds.load(QM9_TEST_DATA_PATH)
        ds_COPY = copy.deepcopy(ds)
        ds.save_state(directory="TEST_DIR", override=False)
        new_ds = QM9Dataset()
        new_ds.load_state("TESTDS", directory="TEST_DIR")

        for key, value in new_ds.__dict__.items():
            if key == 'raw':
                for qm9ID, datum in ds_COPY.raw.items():
                    assert datum.smiles == new_ds[qm9ID].smiles
            else:
                assert getattr(ds_COPY, key) == value

        shutil.rmtree('TEST_DIR')


class TestQM9MLDataset:

    def test_save_load_state(self):
        ds = QM9Dataset(dsname="TESTDS")
        ds.load(QM9_TEST_DATA_PATH)
        mlds = QM9GraphDataset(ds)
        mlds_COPY = copy.deepcopy(mlds)
        mlds.save_state(directory="TEST_DIR", override=False)
        new_mlds = QM9GraphDataset()
        new_mlds.load_state("TESTDS", directory="TEST_DIR")

        for key, value in new_mlds.__dict__.items():
            if key == 'raw':
                for qm9ID, datum in mlds_COPY.raw.items():
                    assert datum.smiles == new_mlds.raw[qm9ID].smiles
            elif key == 'ml_ready':
                pass
            else:
                assert getattr(mlds_COPY, key) == value

        shutil.rmtree('TEST_DIR')

    def test_initial_pipeline(self):
        ds = QM9Dataset(dsname="TESTDS")
        ds.load(QM9_TEST_DATA_PATH)
        mlds = QM9GraphDataset(ds)
        mlds.to_mol()
        mlds.analyze()
        mlds.to_graph()
        mlds.init_ml_data(scale_targets=False)
        mlds.init_ml_data(scale_targets=True, force=True)

    def test_determinism(self):

        ds = QM9Dataset(dsname="TESTDS")
        ds.load(QM9_TEST_DATA_PATH)
        mlds = QM9GraphDataset(ds, seed=54321)
        mlds.to_mol()
        mlds.analyze()
        mlds.to_graph()
        mlds.init_ml_data(scale_targets=True, targets_to_use=None)
        mlds.init_splits()
        loaders1 = mlds.get_loaders()

        np.random.seed(123456)

        ds2 = QM9Dataset(dsname="TESTDS2")
        ds2.load(QM9_TEST_DATA_PATH)
        mlds2 = QM9GraphDataset(ds2, seed=54321)
        mlds2.to_mol()
        mlds2.analyze()
        mlds2.to_graph()
        mlds2.init_ml_data(scale_targets=True, targets_to_use=None)
        mlds2.init_splits()
        loaders2 = mlds2.get_loaders()

        assert mlds.tvt_splits['test'] == mlds2.tvt_splits['test']
        assert mlds.tvt_splits['valid'] == mlds2.tvt_splits['valid']
        assert mlds.tvt_splits['train'] == mlds2.tvt_splits['train']
        assert set(mlds.tvt_splits['test']) \
            .isdisjoint(mlds.tvt_splits['valid'])
        assert set(mlds.tvt_splits['test']) \
            .isdisjoint(mlds.tvt_splits['train'])
        assert set(mlds.tvt_splits['train']) \
            .isdisjoint(mlds.tvt_splits['valid'])

        for batch in loaders1['test']:
            target1 = batch[1]
            break
        for batch in loaders2['test']:
            target2 = batch[1]
            break
        assert np.allclose(target1, target2)

        for batch in loaders1['valid']:
            target1 = batch[1]
            break
        for batch in loaders2['valid']:
            target2 = batch[1]
            break
        assert np.allclose(target1, target2)

        for ii, datum in enumerate(mlds2):
            assert datum[2] == mlds.ml_data[ii][2]


class TestQM9Datum:

    def test_000001(self, ds):
        datum = ds.raw[1]
        assert datum.smiles[0] == 'C'
        assert datum.smiles[1] == 'C'

    def test_100001(self, ds):
        datum = ds.raw[100001]
        assert datum.smiles[0] == 'CCC(C)(C)C(C)C=O'
        assert datum.smiles[1] == 'CCC(C)(C)[C@@H](C)C=O'
