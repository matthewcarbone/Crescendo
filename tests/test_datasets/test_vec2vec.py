#!/usr/bin/env python3

import copy
import numpy as np
import shutil

from crescendo.datasets.vec2vec import Vec2VecDataset


class TestVec2VecDataset:

    def test_save_load_state(self):
        ds = Vec2VecDataset(dsname='test_ds')
        ds.smart_load("data/df_testing_data")
        ds_COPY = copy.deepcopy(ds)
        ds.save_state(directory="TEST_DIR", override=False)
        new_ds = Vec2VecDataset(dsname='test_ds')
        new_ds.load_state(directory="TEST_DIR")

        for key, value in new_ds.__dict__.items():
            if key == 'raw':
                for key2, value2 in value.items():
                    np.allclose(getattr(ds_COPY, key)[key2], value2)
            else:
                getattr(ds_COPY, key) == value

        shutil.rmtree('TEST_DIR')
