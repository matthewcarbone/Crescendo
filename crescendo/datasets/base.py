#!/usr/bin/env python3

import torch

# from crescendo.utils.logger import logger_default as dlog

import torch.utils.data


class _BaseCore(torch.utils.data.Dataset):

    def __init__(self, raise_error=True):
        """Base initializer.

        Parameters
        ----------
        raise_error : bool
            If true, will raise errors and terminate the program on non
            critical errors
        """

        assert isinstance(raise_error, bool)
        self.raise_error = raise_error

        # Step 1: ingestion - this is class-specific
        self.raw = None

        # Step 2: create the features and targets via a featurize method;
        # note that the names are the indexes of each example
        self.features = None
        self.targets = None
        self.names = None

        # Step 3: initialize the machine learning torch-compatible data
        self.ml_data = None

    def __getitem__(self, index):
        return self.ml_data[index]

    def __len__(self):
        return len(self.ml_data)

    def init_ml(self):
        """Initializes the ml_data attribute from the features and targets.
        Takes optional arguments depending on the data type, possibly on how to
        construct the features and targets themselves."""

        raise NotImplementedError


class _AllDataset(_BaseCore):
    """A container for data that contains all data at once: features and
    targets. Generally, classes that inherit this base require both a load
    and featurize step to load in the data from disk and then generate the
    features and targets (featurize step) from the raw data. This ultimately
    produces _Container objects depending on the featurization method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self):
        raise NotImplementedError(
            "Load method is required for _AllDataset"
        )

    def featurize(self):
        raise NotImplementedError(
            "Featurize method is required for _AllDataset"
        )


class _SplitDataset(_BaseCore):
    """A container for data that contains data split into the indexes/names,
    features and targets initially. It requires an input of _Container type
    objects, which wrap different data formats corresponding to features and
    targets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self):
        raise NotImplementedError(
            "Load method is not required for _SplitDataset, "
            "and should not be called"
        )

    def featurize(self):
        raise NotImplementedError(
            "Featurize method is not required for _AllDataset, "
            "and shoult not be called"
        )
