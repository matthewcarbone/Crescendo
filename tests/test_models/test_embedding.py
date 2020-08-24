#!/usr/bin/env python3

import pytest

import torch

from crescendo.models.embedding import Embedding


@pytest.fixture
def dummy_embedding():
    pass


class TestEmbedding:

    def test(self):
        n_classes_per_feature = [4]
        e = Embedding(
            input_dims=n_classes_per_feature,
            embedding_dims=[2]
        )
        classes = torch.tensor([
            [1],
            [2],
            [3],
            [2],
            [3],
            [1]
        ]).long()
        res = e(classes)
        assert res.shape == (6, 2)

    def test_multi_features(self):
        n_classes_per_feature = [4, 5]
        e = Embedding(
            input_dims=n_classes_per_feature,
            embedding_dims=[2, 6]
        )
        classes = torch.tensor([
            [1, 1],
            [2, 4],
            [3, 3],
            [2, 2],
            [3, 4],
            [1, 0]
        ]).long()
        res = e(classes)
        assert res.shape == (6, 8)

    def test_fail(self):
        n_classes_per_feature = [4]
        e = Embedding(
            input_dims=n_classes_per_feature,
            embedding_dims=[2]
        )
        classes = torch.tensor([
            [1],
            [2],
            [3],
            [2],
            [3],
            [4]  # <- this causes the failure
        ]).long()
        with pytest.raises(Exception):
            e(classes)
