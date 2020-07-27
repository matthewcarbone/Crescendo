#!/usr/bin/env python3

import torch

from crescendo.utils.logger import logger_default as dlog


class _BaseCore(torch.utils.data.Dataset):

    def __init__(self, debug=-1, raise_error=True, **kwargs):
        """Base initializer.

        Parameters
        ----------
        debug : int
            If set to -1 (default) then we're not in debug mode. If some
            integer > 0, then that is the number of total data points loaded
            into self.raw.
        raise_error : bool
            If true, will raise errors and terminate the program on non
            critical errors
        """

        self.debug = debug
        self.raise_error = raise_error

    def __getitem__(self, index):
        return self.ml_data[index]

    def __len__(self):
        return len(self.ml_data)

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, d):
        assert isinstance(d, int)
        assert d == -1 or d > 0
        dlog.info(f"Loader debug variable set to {d}")
        self._debug = d

    @property
    def raise_error(self):
        return self._raise_error

    @raise_error.setter
    def raise_error(self, r):
        assert isinstance(r, bool)
        self._raise_error = r

    def load(self):
        """Optional, depending on the loader type. Generally initializes
        raw."""

        raise NotImplementedError

    def featurize(self):
        raise NotImplementedError

    def init_ml(self):
        """Initializes the ml_data attribute from the features and targets.
        Takes optional arguments depending on the data type, possibly on how to
        construct the features and targets themselves."""

        raise NotImplementedError


class _CrescendoAllDataset(_BaseCore):
    """A container for data that contains all data at once: features and
    targets. Generally, classes that inherit this base require both a load
    and featurize step to load in the data from disk and then generate the
    features and targets (featurize step) from the raw data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Step 1: ingestion - this is class-specific
        self.raw = None

        # Step 2: create the features and targets via a featurize method;
        # note that the names are the indexes of each example
        self.features = None
        self.targets = None
        self.names = None

        # Step 3: initialize the machine learning torch-compatible data
        self.ml_data = None


class _CrescendoSplitDataset(_BaseCore):
    """A container for data that contains data split into the indexes/names,
    features and targets initially. Generally, classes that inherit this base
    do not require a load or featurize method as this information is specified
    during initialization."""

    def __init__(
        self, names=None, features=None, targets=None, **kwargs
    ):
        super().__init__(**kwargs)

        # Step 1: ingestion - this is class-specific
        self.raw = None

        # Step 2: create the features and targets via a featurize method;
        # note that the names are the indexes of each example
        self.names = names
        self.features = features
        self.targets = targets

        if any([names is None, features is None, targets is None]):
            critical = \
                f"Names, features and targets (of types {type(names)}, " \
                f"{type(features)}, {type(targets)}) must all be specified" \
                f"for this DataSet class type {self.__class__.__name}"
            dlog.critical(critical)
            raise RuntimeError(critical)

        # Step 3: initialize the machine learning torch-compatible data
        self.ml_data = None
