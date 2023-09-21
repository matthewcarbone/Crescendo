from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def download_california_housing_data(path, random_state=1234):
    path = Path(path)
    if not path.exists():
        path.mkdir()

    housing = fetch_california_housing()

    X_train, X_val, Y_train, Y_val = train_test_split(
        housing["data"],
        housing["target"],
        test_size=0.2,
        random_state=random_state,
    )
    X_test, X_val, Y_test, Y_val = train_test_split(
        X_val,
        Y_val,
        test_size=0.5,
        random_state=random_state,
    )

    np.save(path / "X_train.npy", X_train)
    np.save(path / "X_val.npy", X_val)
    np.save(path / "X_test.npy", X_test)
    np.save(path / "Y_train.npy", Y_train.reshape(-1, 1))
    np.save(path / "Y_val.npy", Y_val.reshape(-1, 1))
    np.save(path / "Y_test.npy", Y_test.reshape(-1, 1))
