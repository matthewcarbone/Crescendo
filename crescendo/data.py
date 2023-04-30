"""Container for various LightningDataModules."""


from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory

from lightning import LightningDataModule
import numpy as np
from rich.console import Console
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from crescendo.utils.datasets import download_california_housing_data
from crescendo.utils.other_utils import read_json


console = Console()


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
        return self._apply_ensemble_split(dat)

    @cached_property
    def X_val(self):
        return self._load_data("X_val")

    @cached_property
    def X_test(self):
        return self._load_data("X_test")

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
        return self._X_scaler.transform(self._X_train)

    @cached_property
    def X_val_scaled(self):
        if self._X_scaler is None:
            raise ValueError("X-scaler is disabled")
        return self._X_scaler.transform(self._X_val)

    @cached_property
    def X_test_scaled(self):
        if self._X_scaler is None:
            raise ValueError("X-scaler is disabled")
        return self._X_scaler.transform(self._X_test)

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
    def _init_dataloader_kwargs(self):
        self._dataloader_kwargs = {
            "batch_size": self.hparams.batch_size,
            "num_workers": self.hparams.num_workers,
            "pin_memory": self.hparams.pin_memory,
        }

    def train_dataloader(self):
        X = self.X_train.copy()
        if self._X_scaler is not None:
            X = self._X_scaler.transform(X)
        X = torch.tensor(X).float()
        Y = torch.tensor(self.Y_train.copy()).float()
        return DataLoader(TensorDataset(X, Y), **self._dataloader_kwargs)

    def val_dataloader(self):
        X = self.X_val.copy()
        if self._X_scaler is not None:
            X = self._X_scaler.transform(X)
        X = torch.tensor(X).float()
        Y = torch.tensor(self.Y_val.copy()).float()
        return DataLoader(TensorDataset(X, Y), **self._dataloader_kwargs)

    def test_dataloader(self):
        X = self.X_test.copy()
        if self._X_scaler is not None:
            X = self._X_scaler.transform(X)
        X = torch.tensor(X).float()
        Y = torch.tensor(self.Y_test.copy()).float()
        return DataLoader(TensorDataset(X, Y), **self._dataloader_kwargs)


class ArrayRegressionDataModule(
    XYArrayPropertyMixin, ScaleXMixin, DataLoaderMixin, LightningDataModule
):
    """A standard data module for array data.

    Notes
    -----
    ``LightningDataModule`` implements the following methods, which we should
    keep in the same order:

    - prepare_data: things done on a single compute unit cpu/gpu, such as
      downloading data, preprocessing, etc.
    - setup: things done on every ddp process
    - train_dataloader
    - val_dataloader
    - test_dataloader
    - teardown: called on every process

    See the documentation for further reference
    https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
        normalize_inputs=True,
        ensemble_split_index=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._setup_X_scaler()
        self._init_dataloader_kwargs()


class CaliforniaHousingDataset(
    XYArrayPropertyMixin, ScaleXMixin, DataLoaderMixin, LightningDataModule
):
    def __init__(
        self,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
        normalize_inputs=True,
    ):
        super().__init__()
        self.hparams.ensemble_split_index = None
        self.hparams.data_dir = None
        self.save_hyperparameters(logger=False)
        with TemporaryDirectory() as t:
            path = t / Path("california_housing_data")
            download_california_housing_data(path)
            self._X_train = np.load(path / "X_train.npy")
            self._X_val = np.load(path / "X_val.npy")
            self._Y_train = np.load(path / "Y_train.npy")
            self._Y_val = np.load(path / "Y_val.npy")
            self._X_test = None
            self._Y_test = None
        self._setup_X_scaler()
        self._init_dataloader_kwargs()
