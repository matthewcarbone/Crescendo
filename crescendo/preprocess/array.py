"""Useful functions for preprocessing array data."""

from pathlib import Path

import numpy as np
from rich.console import Console
from sklearn.model_selection import KFold

from crescendo import utils

console = Console()


def ensemble_split(data_dir, n_splits=20, shuffle=True, random_state=42):
    """Creates an auxiliary file in ``data_dir`` containing the indexes of
    multiple folds for ensemble training via training set downsampling.

    Parameters
    ----------
    data_dir : os.PathLike
        Data directory. Must contain X_train.npy.
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
