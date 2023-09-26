"""Container for various LightningDataModules used for vector-to-vector
predictions."""

from pathlib import Path
from tempfile import TemporaryDirectory

from lightning import LightningDataModule
import numpy as np

from crescendo.utils.datasets import download_california_housing_data
from crescendo.data._common import (
    XYArrayPropertyMixin,
    ScaleXMixin,
    DataLoaderMixin,
)
from crescendo import logger


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
    ensemble_split : dict
        Dictionary with the arguments for crescendo.data.ensemble_split.split.
        By default, it is disabled (see its default config)
    feature_select : str
        If not None, this argument provides some custom functionality to select
        only a subset of the features provided in the data. For example,
        ``feature_select="0:200,400:600"`` will select features 0 through 199,
        inclusive, and 400 through 599, inclusive.
    dataloader_kwargs : dict
        A dictionary containing the keyword arguments to pass to all
        dataloaders.
    """

    def __init__(
        self,
        data_dir,
        normalize_inputs,
        ensemble_split,
        feature_select,
        dataloader_kwargs,
        production_mode,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._setup_X_scaler()
        if self.hparams.production_mode:
            logger.warning(
                "Production mode is set to True. Validation and testing data "
                "will be combined with training data during model fitting."
            )


class CaliforniaHousingDataset(
    XYArrayPropertyMixin, ScaleXMixin, DataLoaderMixin, LightningDataModule
):
    def __init__(
        self,
        normalize_inputs,
        ensemble_split,
        feature_select,
        dataloader_kwargs,
        production_mode,
    ):
        super().__init__()
        self.hparams.data_dir = None
        self.save_hyperparameters(logger=False)
        with TemporaryDirectory() as t:
            path = t / Path("california_housing_data")
            download_california_housing_data(path)
            self._X_train = np.load(path / "X_train.npy")
            self._X_val = np.load(path / "X_val.npy")
            self._X_test = np.load(path / "X_test.npy")
            self._Y_train = np.load(path / "Y_train.npy")
            self._Y_val = np.load(path / "Y_val.npy")
            self._Y_test = np.load(path / "Y_test.npy")
        if self.hparams.production_mode:
            logger.warning(
                "Production mode is set to True. Validation and testing data "
                "will be combined with training data during model fitting."
            )
        self._setup_X_scaler()
