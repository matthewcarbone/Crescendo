from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from crescendo.preprocess.array import split_and_save


def test_split_and_save(X_array, Y_array):
    p_val = 0.123
    p_test = 0.234
    p_train = 1.0 - p_val - p_test
    N = X_array.shape[0]

    with TemporaryDirectory() as d:
        d = Path(d)

        # Save the data
        split_and_save(X_array, Y_array, d, p_val, p_test, random_state=12345)

        # Now load it and check that it has the right proportions
        X_train = np.load(d / "X_train.npy")
        X_val = np.load(d / "X_val.npy")
        X_test = np.load(d / "X_test.npy")
        Y_train = np.load(d / "Y_train.npy")
        Y_val = np.load(d / "Y_val.npy")
        Y_test = np.load(d / "Y_test.npy")

        eps = 1e-8
        assert p_test - eps < X_test.shape[0] / N < p_test + eps
        assert p_test - eps < Y_test.shape[0] / N < p_test + eps
        assert p_val - eps < X_val.shape[0] / N < p_val + eps
        assert p_val - eps < Y_val.shape[0] / N < p_val + eps
        assert p_train - eps < X_train.shape[0] / N < p_train + eps
        assert p_train - eps < Y_train.shape[0] / N < p_train + eps
