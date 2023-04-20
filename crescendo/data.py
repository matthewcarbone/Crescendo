"""
BSD 3-Clause License

Copyright (c) 2022, Brookhaven Science Associates, LLC, Brookhaven National
Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


from functools import cached_property
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule


# Note currently not implemented
def random_downsample(arrays, *, keep_prop=0.9, replace=False, seed=None):
    """Takes an arbitrary number of arrays as input arguments and returns
    randomly downsampled versions.

    Parameters
    ----------
    arrays
        A list of arrays to downsample, all in the same way
    size : None, optional
        Description
    replace : bool, optional
        Description
    p : None, optional
        Description
    seed : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """

    np.random.seed(seed)

    if keep_prop == 1.0:
        return arrays
    elif keep_prop > 1.0 or keep_prop < 0.0:
        raise ValueError(f"keep_prop {keep_prop} must be in 0, 1")
    L = arrays[0].shape[0]  # Number of examples
    choice = np.random.choice(
        L,
        size=int(keep_prop * L),
        replace=replace,
        p=None
    )
    print(
        f"random sampled choice indexes: {choice[:10]} ... LEN={len(choice)}"
    )
    return [arr[choice, ...] for arr in arrays]


class ArrayRegressionData(LightningDataModule):
    """Used for simple array data on disk, nothing fancy. Casts the data to
    float tensors."""

    @cached_property
    def X_train(self):
        return np.load(Path(self._data_dir) / "X_train.npy")

    @cached_property
    def X_val(self):
        return np.load(Path(self._data_dir) / "X_val.npy")

    @cached_property
    def X_test(self):
        path = Path(self._data_dir) / "X_test.npy"
        if not path.exists():
            print(f"{path} does not exist")
            return None
        return np.load(path)

    @cached_property
    def Y_train(self):
        return np.load(Path(self._data_dir) / "Y_train.npy")

    @cached_property
    def Y_val(self):
        return np.load(Path(self._data_dir) / "Y_val.npy")

    @cached_property
    def Y_test(self):
        path = Path(self._data_dir) / "Y_test.npy"
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
        *,
        data_dir="./",
        train_loader_kwargs={
            "batch_size": 64,
            "persistent_workers": True,
            "pin_memory": True,
            "num_workers": 3,
        },
        val_loader_kwargs={
            "batch_size": 64,
            "persistent_workers": True,
            "pin_memory": True,
            "num_workers": 3,
        },
        test_loader_kwargs={
            "batch_size": 64,
            "persistent_workers": True,
            "pin_memory": True,
            "num_workers": 3,
        },
        parallel=False,
        scale_inputs=True,
    ):
        super().__init__()
        
        self._data_dir = data_dir
        self._train_loader_kwargs = train_loader_kwargs
        self._val_loader_kwargs = val_loader_kwargs
        self._test_loader_kwargs = test_loader_kwargs
        
        if parallel:
            self._train_loader_kwargs["multiprocessing_context"] = 'fork'
            self._val_loader_kwargs["multiprocessing_context"] = 'fork'
            self._test_loader_kwargs["multiprocessing_context"] = 'fork'

        if scale_inputs:
            self._scaler = StandardScaler()
            self._scaler.fit(self.X_train)
        else:
            class _dummy:
                def transform(self, x):
                    return x
            self._scaler = _dummy()

    def train_dataloader(self):
        X = self.X_train.copy()
        X = self._scaler.transform(X)
        X = torch.tensor(X).float()
        Y = torch.tensor(self.Y_train.copy()).float()
        return DataLoader(TensorDataset(X, Y), **self._train_loader_kwargs)

    def val_dataloader(self):
        X = self.X_val.copy()
        X = self._scaler.transform(X)
        X = torch.tensor(X).float()
        Y = torch.tensor(self.Y_val.copy()).float()
        return DataLoader(TensorDataset(X, Y), **self._val_loader_kwargs)

    def test_dataloader(self):
        X = self.X_test.copy()
        X = self._scaler.transform(X)
        X = torch.tensor(X).float()
        Y = torch.tensor(self.Y_test.copy()).float()
        return DataLoader(TensorDataset(X, Y), **self._val_loader_kwargs)
