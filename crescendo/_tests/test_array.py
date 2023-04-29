from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from crescendo.preprocess.array import split_and_save


def test_split_and_save(X_array, Y_array):
    with TemporaryDirectory() as d:
        d = Path(d)

        # Save the data
        split_and_save(X_array, Y_array, d, 0.2, 0.2, random_state=12345)

        # Now load it and check that it has the right proportions
        _ = np.load(d / "X_train.npy")
        _ = np.load(d / "X_val.npy")
        X_test = np.load(d / "X_test.npy")
        _ = np.load(d / "Y_train.npy")
        _ = np.load(d / "Y_val.npy")
        Y_test = np.load(d / "Y_test.npy")

        eps = 1e-3
        assert 0.2 - eps < X_test.shape[0] / X_array.shape[0] < 0.2 + eps
        assert 0.2 - eps < Y_test.shape[0] / X_array.shape[0] < 0.2 + eps
