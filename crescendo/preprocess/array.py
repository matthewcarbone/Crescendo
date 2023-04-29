"""Useful functions for preprocessing array data."""

from pathlib import Path

import numpy as np
from rich.console import Console
from sklearn.model_selection import KFold, train_test_split

from crescendo import utils

console = Console()


def split_and_save(X, Y, target, p_valid, p_test, **kwargs):
    """Executes the usual train-validation-test split on provided arrays ``X``
    and ``Y``, then saves the results in the correct format to the provided
    directory. ``p_test`` indicates the proportion of the original data to be
    used for testing. The remainder is used for training and validation.
    ``p_valid`` is the proportion of the remaining training+validation data to
    be used for validation. All data is saved as ``.npy``.

    Parameters
    ----------
    X : numpy.ndarray
    Y : numpy.ndarray
    target : os.PathLike
        The target directory to save the data to.
    p_valid : float
        Proportion of the non-testing data to be used for validation.
    p_test : float
        Proportion of the original data to be used for testing.
    **kwargs
        Extra keyword arguments to be passed to ``train_test_split``.
    """

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=p_test)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=p_valid
    )

    d = Path(target)

    np.save(d / "X_train.npy", X_train)
    np.save(d / "X_val.npy", X_val)
    np.save(d / "X_test.npy", X_test)

    np.save(d / "Y_train.npy", Y_train)
    np.save(d / "Y_val.npy", Y_val)
    np.save(d / "Y_test.npy", Y_test)


def ensemble_split(data_dir, n_splits=20, shuffle=True, random_state=42):
    """Creates an auxiliary file in ``data_dir`` containing the indexes of
    multiple folds for ensemble training via training set downsampling.

    Parameters
    ----------
    data_dir : os.PathLike
        Data directory. Must contain X_train.npy.
    n_splits : int, optional
        The number of KFold splits to take.
    shuffle : bool, optional
    random_state : int, optional
    """

    assert n_splits > 1

    root = Path(data_dir)
    path = root / "X_train.npy"
    X_train = np.load(path)

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    res = {
        f"split-{ii}": train_index.tolist()
        for ii, (train_index, _) in enumerate(kf.split(X_train))
    }
    props = np.array([len(r) / X_train.shape[0] * 100.0 for r in res.values()])
    mu = props.mean()

    console.log(
        f"Splitting training data of shape {X_train.shape} at '{path}' into "
        f"{n_splits} splits using random_state={random_state}. Resulting "
        f"splits constitute {mu:.01f}% of the training data."
    )

    assert set(res["split-0"]) != set(res["split-1"])

    save_path = root / "splits.json"
    utils.save_json(res, save_path)


def construct_example_from_data(data_dir, example_data_dir, keep_prop=0.05):
    """Creates an example of a real database to put on GitHub for use in
    smoke tests and for examples for people.

    Parameters
    ----------
    data_dir : os.PathLike
        The source of the data. Must contain ``X_train.npy``, ``X_val.npy``,
        etc.
    example_data_dir : os.PathLike
        The target for the example.
    keep_prop : float, optional
        The proportion of the data to actually keep for the example.
    """

    target = Path(example_data_dir)
    target.mkdir(exist_ok=True, parents=True)
    data_dir = Path(data_dir)

    # Training data
    X_train = np.load(data_dir / "X_train.npy")
    Y_train = np.load(data_dir / "Y_train.npy")
    _, X, _, Y = train_test_split(X_train, Y_train, test_size=keep_prop)
    console.log(f"New example training data of shapes {X.shape} & {Y.shape}")
    np.save(target / "X_train.npy", X)
    np.save(target / "Y_train.npy", Y)
    # except FileNotFoundError?

    # Validation data
    X_val = np.load(data_dir / "X_val.npy")
    Y_val = np.load(data_dir / "Y_val.npy")
    _, X, _, Y = train_test_split(X_val, Y_val, test_size=keep_prop)
    console.log(f"New example validation data of shapes {X.shape} & {Y.shape}")
    np.save(target / "X_val.npy", X)
    np.save(target / "Y_val.npy", Y)

    # Testing data
    X_test = np.load(data_dir / "X_test.npy")
    Y_test = np.load(data_dir / "Y_test.npy")
    _, X, _, Y = train_test_split(X_test, Y_test, test_size=keep_prop)
    console.log(f"New example testing data of shapes {X.shape} & {Y.shape}")
    np.save(target / "X_test.npy", X)
    np.save(target / "Y_test.npy", Y)

    console.log(f"Data saved to target={target}")
