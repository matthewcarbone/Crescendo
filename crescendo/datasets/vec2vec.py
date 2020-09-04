#!/usr/bin/env python3

import datetime
import os as os
import pandas as pd


import torch

from crescendo.utils.logger import logger_default as dlog


class Vec2VecDataset(torch.utils.data.Dataset):
    """A standard dataset for containing vector-to-vector data. Essentially,
    the loaded features, targets and metadata should all be fixed-lengh
    vectors.

    Attributes
    ----------
    dsname : str
        The name of the dataset.
    raw : dict[np.array]
        A dictionary of numpy arrays pertaining to the features, targets and
        metadata. Initialized by load.
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











