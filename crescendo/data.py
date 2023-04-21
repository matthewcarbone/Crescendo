"""Container for various LightningDataModules
Code is modified based off of 
https://github.com/ashleve/lightning-hydra-template/blob/
89194063e1a3603cfd1adafa777567bc98da2368/src/data/mnist_datamodule.py

MIT License

Copyright (c) 2021 ashleve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class ArrayRegressionDataModule(LightningDataModule):
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
    
    Attributes
    ----------
    transforms : TYPE
        Description
    """

    @cached_property
    def X_train(self):
        return np.load(Path(self.hparams.data_dir) / "X_train.npy")

    @cached_property
    def X_val(self):
        return np.load(Path(self.hparams.data_dir) / "X_val.npy")

    @cached_property
    def X_test(self):
        path = Path(self.hparams.data_dir) / "X_test.npy"
        if not path.exists():
            print(f"{path} does not exist")
            return None
        return np.load(path)

    @cached_property
    def Y_train(self):
        return np.load(Path(self.hparams.data_dir) / "Y_train.npy")

    @cached_property
    def Y_val(self):
        return np.load(Path(self.hparams.data_dir) / "Y_val.npy")

    @cached_property
    def Y_test(self):
        path = Path(self.hparams.data_dir) / "Y_test.npy"
        if not path.exists():
            print(f"{path} does not exist")
            return None
        return np.load(path)

    @property
    def n_features(self):
        return self.X_train.shape[1]

    @property
    def n_targets(self):
        return self.Y_train.shape[1]

    def __init__(
        self,
        data_dir,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
        normalize_inputs=True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """
        
        Parameters
        ----------
        stage : Optional[str], optional
            Description
        """

        if self.hparams.normalize_inputs:
            self._scaler = StandardScaler()
            self._scaler.fit(self.X_train)
        else:
            self._scaler = None

        self._dataloader_kwargs = {
            "batch_size": self.hparams.batch_size,
            "num_workers": self.hparams.num_workers,
            "pin_memory": self.hparams.pin_memory,
        }

    def train_dataloader(self):
        X = self.X_train.copy()
        if self._scaler is not None:
            X = self._scaler.transform(X)
        X = torch.tensor(X).float()
        Y = torch.tensor(self.Y_train.copy()).float()
        return DataLoader(TensorDataset(X, Y), **self._dataloader_kwargs)

    def val_dataloader(self):
        X = self.X_val.copy()
        if self._scaler is not None:
            X = self._scaler.transform(X)
        X = torch.tensor(X).float()
        Y = torch.tensor(self.Y_val.copy()).float()
        return DataLoader(TensorDataset(X, Y), **self._dataloader_kwargs)

    def test_dataloader(self):
        X = self.X_test.copy()
        if self._scaler is not None:
            X = self._scaler.transform(X)
        X = torch.tensor(X).float()
        Y = torch.tensor(self.Y_test.copy()).float()
        return DataLoader(TensorDataset(X, Y), **self._dataloader_kwargs)
