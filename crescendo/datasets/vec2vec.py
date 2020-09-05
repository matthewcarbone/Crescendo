#!/usr/bin/env python3

import datetime
import numpy as np
import os as os
import pandas as pd
import pickle
import random


import torch

from crescendo.defaults import VEC2VEC_GENERAL_DS_ENV_VAR, P_PROTOCOL
from crescendo.utils.py_utils import check_for_environment_variable
from crescendo.utils.logger import logger_default as dlog
from crescendo.utils.timing import time_func
from crescendo.samplers.base import Sampler as RandomSampler


class _SimpleLoadingAndSaving:

    def save_state(self, directory=None, override=False):
        """Dumps the class as a dictionary into a pickle file. It will save
        the object to the dsname directory as raw.pkl. Specifically, it
        saves to directory/dsname/mld.pkl. For this particular class, we skip
        the raw datatype, since this class will double as the machine
        learning-ready dataset as well."""

        if directory is None:
            directory = check_for_environment_variable(
                VEC2VEC_GENERAL_DS_ENV_VAR
            )

        full_dir = f"{directory}/{self.dsname}"
        full_path = f"{full_dir}/mld.pkl"

        if os.path.exists(full_path) and not override:
            error = \
                f"Full path {full_path} exists and override is False - " \
                "exiting and not overwriting"
            dlog.error(error)
            return

        elif os.path.exists(full_path) and override:
            warning = \
                f"Full path {full_path} exists and override is True - " \
                "overwriting saved dataset"
            dlog.warning(warning)

        os.makedirs(full_dir, exist_ok=True)

        d = self.__dict__
        pickle.dump(d, open(full_path, 'wb'), protocol=P_PROTOCOL)

        dlog.info(f"Saved {full_path}")

    def load_state(self, directory=None):
        """Reloads the dataset of the specified name and directory."""

        if directory is None:
            directory = check_for_environment_variable(
                VEC2VEC_GENERAL_DS_ENV_VAR
            )

        full_path = f"{directory}/{self.dsname}/mld.pkl"
        d = pickle.load(open(full_path, 'rb'))

        for key, value in d.items():
            setattr(self, key, value)

        dlog.info(f"Loaded from {full_path}")

    def check_exists(self, directory=None):
        if directory is None:
            directory = check_for_environment_variable(
                VEC2VEC_GENERAL_DS_ENV_VAR
            )

        full_dir = f"{directory}/{self.dsname}"
        full_path = f"{full_dir}/mld.pkl"
        if os.path.exists(full_path):
            return full_path
        return None


class Vec2VecDataset(torch.utils.data.Dataset, _SimpleLoadingAndSaving):
    """A standard dataset for containing vector-to-vector data. Essentially,
    the loaded features, targets and metadata should all be fixed-lengh
    vectors.

    Attributes
    ----------
    dsname : str
        The name of the dataset.
    raw : dict[np.array]
        A dictionary of numpy arrays pertaining to the features, targets and
        metadata. Initialized by load. Data loaded in should generally be saved
        via something like df.to_csv(f'{fname}.csv', index_label=False).
    ml_data : list
        A list of the data as prepared to be processed by a collating function.
        Essentially data that is ready for pytorch to take over.
    """

    def __init__(self, dsname, debug=-1, seed=None):
        self.dt_created = datetime.datetime.now()
        self.dsname = dsname
        self.debug = debug

        self.feature_metadata = None
        self.target_metadata = None
        self.tvt_splits = None

        self.seed = seed

        self.raw = None
        self.ml_data = None

        self.n_features = None
        self.n_targets = None

    def __getitem__(self, ii):
        return self.ml_data[ii]

    def __len__(self):
        return len(self.raw)

    def load(
        self, feature_path, target_path, meta_path=None, index_path=None
    ):
        """Populates the raw attribute by loading in .csv data with headers
        from the provided three paths. Note that the data must be separated
        like this before using load. This ensures no ambiguity about which
        columns correspond to features, targets or metadata. The user must have
        made those distinctions before the pipeline begins.

        Parameters
        ----------
        feature_path, target_path : str
            Required paths to the features and targets.
        meta_path, index_path : str, optional
            Specifies the metadata, if desired.
        """

        nrows = None if self.debug == -1 else self.debug

        if feature_path is None or target_path is None:
            critical = "Feature and target paths must both be specified."
            dlog.critical(critical)
            raise RuntimeError(critical)

        dlog.info(f"Reading features from {feature_path}")
        feat = pd.read_csv(feature_path, nrows=nrows)

        dlog.info(f"Reading targets from {target_path}")
        trg = pd.read_csv(target_path, nrows=nrows)

        if meta_path is not None:
            dlog.info(f"Reading metadata from {meta_path}")
        meta = pd.read_csv(meta_path, nrows=nrows) if meta_path is not None \
            else None

        if index_path is not None:
            dlog.info(f"Reading indexes from {index_path}")
        idx = pd.read_csv(index_path, nrows=nrows) if index_path is not None \
            else [ii for ii in range(len(feat.index))]
        if not isinstance(idx, list):
            idx = idx.to_numpy().flatten()

        # Assert that the indexes of the features and targets are the same
        assert set(feat.index) == set(trg.index)

        # Do the same for the metadata if it is not None
        if meta is not None:
            assert set(feat.index) == set(meta.index)

        self.raw = {
            'features': feat.to_numpy(),
            'targets': trg.to_numpy(),
            'meta': meta.to_numpy() if meta is not None else None,
            'idx': idx
        }

        assert len(self.raw['features']) == len(self.raw['targets']) \
            == len(self.raw['meta']) == len(self.raw['idx'])

        fshape = self.raw['features'].shape
        tshape = self.raw['targets'].shape
        dlog.info(f"Done: features {fshape} targets {tshape}")
        self.n_features = self.raw['features'].shape[1]
        self.n_targets = self.raw['targets'].shape[1]

        if meta is not None:
            mshape = self.raw['meta'].shape
            dlog.info(f"Also read metadata {mshape}")

    def smart_load(self, directory):
        """Attempts to find files that correspond to features, targets,
        metadata and indexes by looking for file keywords in the specified
        directory.

        Parameters
        ----------
        directory : str
            The target directory
        """

        paths = {
            'feature_path': None,
            'target_path': None,
            'meta_path': None,
            'index_path': None
        }

        directory_paths = os.listdir(directory)
        directory_paths = [os.path.join(directory, d) for d in directory_paths]
        directory_paths = [d for d in directory_paths if not os.path.isdir(d)]

        for p in directory_paths:
            base = os.path.basename(p)

            if "feature" in base and ".csv" in base:
                if paths['feature_path'] is None:
                    dlog.info(f"Found feature data at {p}")
                    paths['feature_path'] = p

            elif "target" in base and ".csv" in base:
                if paths['target_path'] is None:
                    dlog.info(f"Found target data at {p}")
                    paths['target_path'] = p

            elif "meta" in base and ".csv" in base:
                if paths['meta_path'] is None:
                    dlog.info(f"Found meta data at {p}")
                    paths['meta_path'] = p

            elif ("index" in base or "idx" in base) and ".csv" in base:
                if paths['index_path'] is None:
                    dlog.info(f"Found index data at {p}")
                    paths['index_path'] = p

        if paths['feature_path'] is None and paths['target_path'] is None:
            error = \
                f"Smart loading failed, check directory {directory} - " \
                "exiting without loading."
            dlog.error(error)
            return

        self.load(**paths)

    def init_splits(
        self, p_tvt=(0.1, 0.1, None), method='random', force=False,
        splits_override=None
    ):
        """Chooses the splits based on some criteria. Note that the first time
        this method is called, the attribute tvt_splits will be set, but it
        will not allow the user to rewrite that split unless force=True. This
        is a failsafe mechanism to prevent bias in the data by constantly
        reshuffling the splits while evaluating the results.

        Parameters
        ----------
        p_tvt : tuple
            The proportions of the data to use in the testing, validation
            and training splits. Note that the last element of the tuple can
            be None, which means to use the remainder of the data in the
            training set after the proportions have been specified for testing
            and validation. In the default case, (0.1, 0.1, None), that means
            that 80% of the data will be used for training.
        method : {'random'}
            The protocol for creating the splits. Currently only a random
            sampler is implemented.
        force : bool
            If Force is False and the splits are already initialized, do
            nothing and log an error. Default is False.
        splits_override : dict, optional
            A dictionary with keys train, valid, test containing lists with the
            indexes of the splits. Default is None, and if specified, it will
            initialize the splits to these values.
        """

        if self.tvt_splits is not None:
            dlog.error("init_splits already initialized and force is False")
            if force:
                dlog.warning(
                    "You already called init_splits but force is True: "
                    "re-running"
                )
            else:
                dlog.error(
                    "You already called init_splits and force is False: "
                    "exiting without re-running"
                )
                return

        if splits_override is not None:
            self.tvt_splits = splits_override
            assert set(self.tvt_splits['train']) \
                .isdisjoint(self.tvt_splits['valid'])
            assert set(self.tvt_splits['train']) \
                .isdisjoint(self.tvt_splits['test'])
            assert set(self.tvt_splits['test']) \
                .isdisjoint(self.tvt_splits['valid'])
            dlog.info("Using the user-specified splits override")
            return

        np.random.seed(self.seed)

        if method == 'random':
            s = RandomSampler(len(self.raw['features']))
            s.shuffle_(self.seed)
            assert s.indexes_modified
            self.tvt_splits = s.split(p_tvt[0], p_tvt[1], p_train=p_tvt[2])
        else:
            critical = f"Invalid split method {method}"
            dlog.critical(critical)
            raise NotImplementedError(critical)

    @staticmethod
    def collating_function_vector_to_vector(batch):
        """Collates the vector-to-vector batches in which single training
        examples consist of (features, targets, idxs, metadata)."""

        meta = [xx[3] for xx in batch]
        idxs = [xx[2] for xx in batch]
        targets = torch.tensor([xx[1] for xx in batch])
        features = torch.tensor([xx[0] for xx in batch])

        return (features, targets, idxs, meta)

    def _statistics(self, what, split):
        """Get's the statistics on what (features, targets, ...) for a
        specified split as cached in the tvt_splits."""

        arr = self.raw[what][self.tvt_splits[split], :]
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0)
        _mu = mu.mean()
        _sd = sd.mean()
        dlog.info(f"{what} | {split} statistics: {_mu:.02g} +/- {_sd:.02g}")
        return mu, sd

    @time_func(dlog)
    def init_ml_data(
        self, scale_features=False, scale_targets=False, force=False
    ):
        """Initializes the ml_data attribute by pairing graphs with potential
        targets. The ml_data attribute contains three entries: [feature,
        target, id/metadata]. This method assumes that all data loaded in as
        feature/target pairs are to be used for ML training. Note that for this
        dataset, we require that the splits have been determined.

        Parameters
        ----------
        scale_features, scale_targets : bool
            This provides the option of scaling the features/targets by the
            statistics of the training data. Note that any pre-scaling
            transforms must be performed prior to loading the data via this
            class.
        force : bool
            Override failsaves on e.g. overwriting existing ml_data.
        """

        if self.ml_data is not None:
            dlog.error("ml_data already initialized and force is False")
            if force:
                dlog.warning(
                    "You already called init_ml_data but force is True: "
                    "re-running"
                )
            else:
                dlog.error(
                    "You already called init_ml_data and force is False: "
                    "exiting without re-running"
                )
                return

        np.random.seed(self.seed)
        random.seed(self.seed)

        dlog.info(f"Scaling features: {scale_features}")
        dlog.info(f"Scaling targets: {scale_targets}")

        if scale_features:
            mu, sd = self._statistics('features', 'train')
            self.feature_metadata = (mu, sd)
            self._statistics('features', 'valid')
            self._statistics('features', 'test')
            self.raw['features'] = (self.raw['features'] - mu) / sd

            # This just logs the new statistics for conveinence
            self._statistics('features', 'train')
            self._statistics('features', 'valid')
            self._statistics('features', 'test')

        if scale_targets:
            mu, sd = self._statistics('targets', 'train')
            self.target_metadata = (mu, sd)
            self._statistics('targets', 'valid')
            self._statistics('targets', 'test')
            self.raw['targets'] = (self.raw['targets'] - mu) / sd

            # This just logs the new statistics for conveinence
            self._statistics('targets', 'train')
            self._statistics('targets', 'valid')
            self._statistics('targets', 'test')

        self.ml_data = [
            [
                list(self.raw['features'][ii]),
                list(self.raw['targets'][ii]),
                self.raw['idx'][ii],
                list(self.raw['meta'][ii]) if self.raw['meta'] is not None
                else None
            ] for ii in range(len(self.raw['features']))
        ]

        dlog.info(f"Done init ml_data of size {len(self.ml_data)}")

    def get_loaders(self, batch_sizes=(32, 32, 32)):
        """Returns the loaders as computed by the prior sampling."""

        # Initialize the subset objects
        testSubset = torch.utils.data.Subset(self, self.tvt_splits['test'])
        validSubset = torch.utils.data.Subset(self, self.tvt_splits['valid'])
        trainSubset = torch.utils.data.Subset(self, self.tvt_splits['train'])

        # Initialize the loader objects
        testLoader = torch.utils.data.DataLoader(
            testSubset, batch_size=batch_sizes[0], shuffle=False,
            collate_fn=Vec2VecDataset.collating_function_vector_to_vector
        )
        validLoader = torch.utils.data.DataLoader(
            validSubset, batch_size=batch_sizes[1], shuffle=False,
            collate_fn=Vec2VecDataset.collating_function_vector_to_vector
        )
        trainLoader = torch.utils.data.DataLoader(
            trainSubset, batch_size=batch_sizes[2], shuffle=True,
            collate_fn=Vec2VecDataset.collating_function_vector_to_vector
        )

        return {
            'test': testLoader,
            'valid': validLoader,
            'train': trainLoader
        }
