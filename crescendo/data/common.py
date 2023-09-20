"""Container for various common LightningDataModules mix-ins and other
things."""

from functools import cached_property
from pathlib import Path

import numpy as np
from rich.console import Console
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from crescendo.utils.other_utils import read_json

console = Console()


def check_batch_size(L_set, batch_size):
    """There are some bugs (https://stackoverflow.com/a/49035538) that occur
    when you attempt to use set sizes that are (significantly?) smaller than
    the requrested batch size. This function checks for that.

    Parameters
    ----------
    L_set : int
        The length of the set provided.
    batch_size : int
        The batch size
    """

    if L_set < batch_size:
        raise ValueError(
            "The set size is smaller than the batch size. This is known to "
            "cause weird problems (https://stackoverflow.com/a/49035538). "
            f"Please set the batch size to at most {L_set} or get more data."
        )


class XYArrayPropertyMixin:
    def _apply_ensemble_split(self, data):
        if self.hparams.ensemble_split_index is None:
            return data
        s = f"split-{self.hparams.ensemble_split_index}"
        split = self.ensemble_splits[s]
        new_dat = data[split, :]
        console.log(
            f"Training data downsampled from {data.shape} -> {new_dat.shape}",
            style="bold yellow",
        )
        return new_dat

    def _apply_feature_selection_logic(self, data):
        """Runs the feature selection logic on the data based on the provided
        value of self.hparams.feature_select."""

        if self.hparams.feature_select is None:
            return data

        # First, split by the comma separation
        split = self.hparams.feature_select.split(",")

        # For each entry, we split by the second delimiter, a :
        split = [np.array(xx.split(":")).astype(int).tolist() for xx in split]

        # Edge case
        if len(split) == 1:
            return data[:, split[0][0] : split[0][1]]

        # Otherwise we iterate through the splits and concatenate
        return np.concatenate([data[:, s[0] : s[1]] for s in split], axis=1)

    def _load_data(self, property_name):
        # Attempt to load from disk
        fname = f"{property_name}.npy"
        if self.hparams.data_dir is not None:
            return np.load(Path(self.hparams.data_dir) / fname)

        # If appropriate file does not exist, attempt to access the internal
        # value of the object
        attr_name = f"_{property_name}"
        dat = getattr(self, attr_name)
        if dat is not None:
            return dat

        # Otherwise raise an error
        raise ValueError(f"_{attr_name} not initialized and dne on disk")

    @cached_property
    def ensemble_splits(self):
        path = Path(self.hparams.data_dir) / "splits.json"
        if not path.exists():
            return None
        return read_json(path)

    @cached_property
    def X_train(self):
        dat = self._load_data("X_train")
        dat = self._apply_ensemble_split(dat)
        return self._apply_feature_selection_logic(dat)

    @cached_property
    def X_val(self):
        dat = self._load_data("X_val")
        return self._apply_feature_selection_logic(dat)

    @cached_property
    def X_test(self):
        dat = self._load_data("X_test")
        return self._apply_feature_selection_logic(dat)

    @cached_property
    def Y_train(self):
        dat = self._load_data("Y_train")
        return self._apply_ensemble_split(dat)

    @cached_property
    def Y_val(self):
        return self._load_data("Y_val")

    @cached_property
    def Y_test(self):
        return self._load_data("Y_test")

    @cached_property
    def n_features(self):
        return self.X_train.shape[1]

    @cached_property
    def n_targets(self):
        return self.Y_train.shape[1]


class ScaleXMixin:
    @cached_property
    def X_train_scaled(self):
        if self._X_scaler is None:
            raise ValueError("X-scaler is disabled")
        return self._X_scaler.transform(self.X_train)

    @cached_property
    def X_val_scaled(self):
        if self._X_scaler is None:
            raise ValueError("X-scaler is disabled")
        return self._X_scaler.transform(self.X_val)

    @cached_property
    def X_test_scaled(self):
        if self._X_scaler is None:
            raise ValueError("X-scaler is disabled")
        return self._X_scaler.transform(self.X_test)

    def _setup_X_scaler(self):
        """

        Parameters
        ----------
        stage : Optional[str], optional
            Description
        """

        if self.hparams.normalize_inputs:
            self._X_scaler = StandardScaler()
            self._X_scaler.fit(self.X_train)
        else:
            self._X_scaler = None


class DataLoaderMixin:
    def train_dataloader(self):
        X = self.X_train.copy()
        check_batch_size(len(X), self.hparams.dataloader_kwargs["batch_size"])
        if self._X_scaler is not None:
            X = self._X_scaler.transform(X)
        X = torch.tensor(X).float()
        Y = torch.tensor(self.Y_train.copy()).float()
        return DataLoader(
            TensorDataset(X, Y), **self.hparams.dataloader_kwargs
        )

    def val_dataloader(self):
        X = self.X_val.copy()
        check_batch_size(len(X), self.hparams.dataloader_kwargs["batch_size"])
        if self._X_scaler is not None:
            X = self._X_scaler.transform(X)
        X = torch.tensor(X).float()
        Y = torch.tensor(self.Y_val.copy()).float()
        return DataLoader(
            TensorDataset(X, Y), **self.hparams.dataloader_kwargs
        )

    def test_dataloader(self):
        X = self.X_test.copy()
        check_batch_size(len(X), self.hparams.dataloader_kwargs["batch_size"])
        if self._X_scaler is not None:
            X = self._X_scaler.transform(X)
        X = torch.tensor(X).float()
        Y = torch.tensor(self.Y_test.copy()).float()
        return DataLoader(
            TensorDataset(X, Y), **self.hparams.dataloader_kwargs
        )
