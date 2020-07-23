#!/usr/bin/env python3

import datetime
from pytz import timezone

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base class for containing a data set read from disk.

    Attributes
    ----------
    raw
        Loaded data in a general format.
    ml_data : list
        List of the entire dataset as ready to be processed by torch's ML
        pipeline (Dataset -> DataLoader).
    init_time : datetime.datetime
        The time at which the dataset object was initialized. Timezone is UTC.
    """

    def __init__(self, name=None):
        """Initializer.

        Parameters
        ----------
        name : str, optional
            The user has the option of naming the dataset.
        """

        self.raw = None
        self.ml_data = None
        self.name = name
        self.init_time = datetime.datetime.now(timezone('UTC'))

    def __getitem__(self, index):
        return self.ml_data[index]

    def load(self):
        raise NotImplementedError

    def ml_ready(self):
        raise NotImplementedError
