import pytest

import numpy as np


@pytest.fixture
def X_array():
    np.random.seed(123)
    return np.random.random((1000, 3))


@pytest.fixture
def Y_array():
    np.random.seed(124)
    return np.random.random((1000, 2))


@pytest.fixture
def X_array_int_cols():
    # 100 x 12 array
    return np.array([[ii] * 100 for ii in range(12)]).T


@pytest.fixture
def Y_array_int_cols():
    # 100 x 3 array
    return np.array([[ii] * 100 for ii in range(3)]).T
