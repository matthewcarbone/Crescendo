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
