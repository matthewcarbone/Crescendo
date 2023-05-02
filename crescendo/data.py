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
        dat = self._load_data("Y_val")
        return self._apply_feature_selection_logic(dat)

    @cached_property
    def Y_test(self):
        dat = self._load_data("Y_test")
        return self._apply_feature_selection_logic(dat)

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
    def train_dataloader(self):
        X = self.X_train.copy()
        if self._X_scaler is not None:
            X = self._X_scaler.transform(X)
        X = torch.tensor(X).float()
        Y = torch.tensor(self.Y_train.copy()).float()
        return DataLoader(
            TensorDataset(X, Y), **self.hparams.dataloader_kwargs
        )

    def val_dataloader(self):
        X = self.X_val.copy()
        if self._X_scaler is not None:
            X = self._X_scaler.transform(X)
        X = torch.tensor(X).float()
        Y = torch.tensor(self.Y_val.copy()).float()
        return DataLoader(
            TensorDataset(X, Y), **self.hparams.dataloader_kwargs
        )

    def test_dataloader(self):
        X = self.X_test.copy()
        if self._X_scaler is not None:
            X = self._X_scaler.transform(X)
        X = torch.tensor(X).float()
        Y = torch.tensor(self.Y_test.copy()).float()
        return DataLoader(
            TensorDataset(X, Y), **self.hparams.dataloader_kwargs
        )


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

    Properties
    ----------
    data_dir : os.PathLike
        The location of the data to load. This data is assumed to be kept in
        files like ``X_train.npy``, ``Y_val.npy``, etc.
    normalize_inputs : bool
        If True, will standard-scale the input features to the standard normal
        distribution using the training set, and execute that transform on the
        other splits. These objects are stored in ``X_train_scaled``, etc.
    ensemble_split_index : int, optional
        If not None, this is an integer referencing the file splits.json in the
        same directory as the data. This json file contains keys such as
        "split-0", "split-1", etc., with values corresponding to the training
        set indexes of the split. You can use
        ``crescendo.preprocess.array:ensemble_split`` to create this file.
    feature_select : str, optional
        If not None, this argument provides some custom functionality to select
        only a subset of the features provided in the data. For example,
        ``feature_select="0:200,400:600"`` will select features 0 through 199,
        inclusive, and 400 through 599, inclusive.
    dataloader_kwargs : dict, optional
        A dictionary containing the keyword arguments to pass to all
        dataloaders.
    """

    def __init__(
        self,
        data_dir,
        normalize_inputs=True,
        ensemble_split_index=None,
        feature_select=None,
        dataloader_kwargs={
            "batch_size": 64,
            "num_workers": 0,
            "pin_memory": False,
        },
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._setup_X_scaler()


class CaliforniaHousingDataset(
    XYArrayPropertyMixin, ScaleXMixin, DataLoaderMixin, LightningDataModule
):
    def __init__(
        self,
        normalize_inputs=True,
        dataloader_kwargs={
            "batch_size": 64,
            "num_workers": 0,
            "pin_memory": False,
        },
    ):
        super().__init__()
        self.hparams.ensemble_split_index = None
        self.hparams.data_dir = None
        self.hparams.feature_select = None
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
