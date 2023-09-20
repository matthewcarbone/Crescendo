"""Container for various LightningDataModules used for vector-to-vector
predictions."""

from pathlib import Path
from tempfile import TemporaryDirectory

from lightning import LightningDataModule
import numpy as np
from rich.console import Console

from crescendo.utils.datasets import download_california_housing_data

from crescendo.data._common import (
    XYArrayPropertyMixin,
    ScaleXMixin,
    DataLoaderMixin,
)


console = Console()


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
        normalize_inputs,
        ensemble_split_index,
        feature_select,
        dataloader_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._setup_X_scaler()


class CaliforniaHousingDataset(
    XYArrayPropertyMixin, ScaleXMixin, DataLoaderMixin, LightningDataModule
):
    def __init__(
        self,
        normalize_inputs,
        dataloader_kwargs,
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
