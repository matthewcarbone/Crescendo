from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from crescendo.data.array import ArrayRegressionDataModule


class TestArrayRegressionDataModule:
    def test_ArrayRegressionDataModule(
        self, X_array_int_cols, Y_array_int_cols
    ):
        # Save the dummy data to disk
        with TemporaryDirectory() as d:
            np.save(str(Path(d) / "X_train.npy"), X_array_int_cols)
            np.save(str(Path(d) / "Y_train.npy"), Y_array_int_cols)

            datamodule = ArrayRegressionDataModule(
                data_dir=d,
                normalize_inputs=False,
                feature_select="0:5",
                ensemble_split={"enable": False},
                dataloader_kwargs={
                    "batch_size": 64,
                    "num_workers": 0,
                    "pin_memory": False,
                    "drop_last": True,
                },
                production_mode=False,
            )
            assert datamodule.X_train.shape[1] == 5
            for ii, column in enumerate(datamodule.X_train.T):
                assert column[0] == ii

            datamodule = ArrayRegressionDataModule(
                data_dir=d,
                normalize_inputs=False,
                feature_select="0:3,6:9",
                ensemble_split={"enable": False},
                dataloader_kwargs={
                    "batch_size": 64,
                    "num_workers": 0,
                    "pin_memory": False,
                    "drop_last": True,
                },
                production_mode=False,
            )

            assert datamodule.X_train.shape[1] == 6
            vals = [0, 1, 2, 6, 7, 8]
            for ii, column in enumerate(datamodule.X_train.T):
                assert column[0] == vals[ii]
