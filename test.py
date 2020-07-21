#!/usr/bin/env python3

import inspect
import pytest
import numpy as np

from crescendo.samplers.base import Sampler
from crescendo.loaders.base import BaseDataset
from crescendo.loaders.text_loaders import CSVDataSet

from crescendo.utils.logger import logger_default


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


class TestSampler:

    def test_init(self):
        """Tests the initialization of the Sampler base class."""

        s = Sampler(10)
        assert np.all(s.indexes == np.array([ii for ii in range(10)]))
        assert not s.indexes_modified

        s = Sampler(100)
        assert np.all(s.indexes == np.array([ii for ii in range(100)]))
        assert not s.indexes_modified

    def test_shuffle_(self):
        """Tests that the indexes are properly shuffled when the shuffle_
        method is called."""

        # 1) Ensure seeding works properly
        s1 = Sampler(10)
        s2 = Sampler(10)
        s1.shuffle_(seed=1234)
        s2.shuffle_(seed=1234)
        assert np.all(s1.indexes == s2.indexes)
        assert s1.indexes_modified
        assert s2.indexes_modified

        # 2) Check no seeding works properly
        s1 = Sampler(1000)
        s2 = Sampler(1000)
        s1.shuffle_()
        s2.shuffle_()
        assert not np.all(s1.indexes == s2.indexes)

    def test_split(self):
        """Critical test: ensures that the T/V/T split results in no
        overlapping datapoints."""

        s = Sampler(1000)
        s.shuffle_()
        tvt_split = s.split(p_test=0.2, p_valid=0.2)
        assert set(tvt_split['test']).isdisjoint(tvt_split['valid'])
        assert set(tvt_split['valid']).isdisjoint(tvt_split['train'])
        assert set(tvt_split['train']).isdisjoint(tvt_split['test'])
        assert not set(tvt_split['test']).isdisjoint(tvt_split['test'])
        assert len(tvt_split['test']) + len(tvt_split['valid']) \
            + len(tvt_split['train']) == 1000

    def test_split_downsample(self):
        """Ensures that explicitly setting a value for p_train during split
        results in the correct behavior."""

        s1 = Sampler(1000)
        s1.shuffle_(seed=1234)
        tvt_split_1 = s1.split(p_test=0.2, p_valid=0.2, p_train=0.4)
        s2 = Sampler(1000)
        s2.shuffle_(seed=1234)
        tvt_split_2 = s2.split(p_test=0.2, p_valid=0.2, p_train=0.5)
        assert np.all(tvt_split_1['test'] == tvt_split_2['test'])
        assert np.all(tvt_split_1['valid'] == tvt_split_2['valid'])
        assert set(tvt_split_1['train']).issubset(tvt_split_2['train'])
        assert not set(tvt_split_2['train']).issubset(tvt_split_1['train'])
        assert len(tvt_split_1['train']) < len(tvt_split_2['train'])

    def test_split_raises_large_props(self):
        """Ensures that the proper RuntimeError is thrown when the user tries
        to specifiy a large enough p_train (or any of the proportions) such
        that the sum of the proportions is greater than 1. This is actually
        quite important since it can lead to undesired behavior in the list
        comprehension if the error is not raised."""

        s1 = Sampler(1000)
        s1.shuffle_()
        with pytest.raises(RuntimeError):
            _ = s1.split(p_test=0.2, p_valid=0.2, p_train=0.61)

        s1 = Sampler(1000)
        s1.shuffle_()
        with pytest.raises(RuntimeError):
            _ = s1.split(p_test=0.21, p_valid=0.2, p_train=0.6)

        s1 = Sampler(1000)
        s1.shuffle_()
        with pytest.raises(RuntimeError):
            _ = s1.split(p_test=0.81, p_valid=0.2)

    def test_mutability(self):
        """Once the T/V/T splits are set, they cannot be changed."""

        s = Sampler(1000)
        s.shuffle_(seed=1234)
        tvt_split = s.split(p_test=0.2, p_valid=0.2, p_train=0.4)
        with pytest.raises(TypeError):
            tvt_split['train'][0] = 12345


def run_all_methods(obj):
    attrs = (getattr(obj, name) for name in dir(obj))
    methods = filter(inspect.ismethod, attrs)
    for method in methods:
        method()


if __name__ == '__main__':

    logger_default.disabled = True
    run_all_methods(TestDataset())
    run_all_methods(TestSampler())
    logger_default.disabled = False
