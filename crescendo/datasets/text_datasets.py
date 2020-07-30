#!/usr/bin/env python3

import torch

from crescendo.datasets.base import _SplitDataset
from crescendo.datum.base import BaseDatum

# from crescendo.utils.logger import logger_default as dlog


class CSVDataset(_SplitDataset):
    """A simple dataset designed to initialize a ML database from csv
    features and targets."""

    def __init__(self, features, targets, names=None):
        """Initializes the CSVLoader object with pd.DataFrame features and
        targets that the user loads beforehand.

        Parameters
        ----------
        features, targets : PandasContainer, ArrayContainer
            Note that the features and targets may contain different numbers
            of columns, but must have the same number of rows (data points).
            Each column corresponds to a different property of a single data
            point which may be a feature or target.
        name : iterable, optional
            A list or pd.Series that contains the indexes of the data points.
            If None, will initialize a list of integers of the length of the
            number of total data points. Default is None.
        """

        # Initializes raw = None, names, features, targets (those three
        # are required), and ml_data = None
        super().__init__()

        self.features = features
        self.targets = targets

        N_samples = self.features.shape[0]
        assert self.targets.shape[0] == N_samples
        self.names = \
            names if names is not None else [ii for ii in range(N_samples)]

    def init_ml(self):
        """Generates the ml_data object."""

        self.ml_data = [
            BaseDatum(
                name=self.names[ii], feature=self.features.iloc[ii],
                target=self.targets.iloc[ii]
            ) for ii in range(len(self.features.index))
        ]

    @staticmethod
    def collating_function(batch):
        """Properly converts the DataSet objects into ones the DataLoader can
        use. The batch is a list of Datum objects, which is iterated over and
        concatenated."""

        return BaseDatum(
            [sub_batch.name for sub_batch in batch],
            torch.tensor([sub_batch.feature for sub_batch in batch]),
            torch.tensor([sub_batch.target for sub_batch in batch])
        )