#!/usr/bin/env python3


import os

import numpy as np
import pandas as pd
import yaml

from crescendo.utils.managers.manager_base import Manager

from crescendo.datasets.vec2vec import Vec2VecDataset
from crescendo.protocols.vec2vec_protocols import Vec2VecProtocol
from crescendo.defaults import VEC2VEC_GENERAL_DS_ENV_VAR
from crescendo.utils.logger import logger_default as dlog
from crescendo.utils.ml_utils import _call_subprocess
from crescendo.utils.py_utils import check_for_environment_variable
from crescendo.utils.ml_utils import save_caches


class Vec2VecManager(Manager):

    def __init__(self, dsname, cache):

        if cache is None:
            cache = check_for_environment_variable(VEC2VEC_GENERAL_DS_ENV_VAR)

        # Location of the cache containing the datasets
        self.root_above = f"{cache}/{dsname}"
        self.dsname = dsname
        self.cache = cache

    def init_ml(self, args):
        """Runs the initialization protocol for creating the raw dataset,
        creating the machine learning-ready attributes from that raw dataset,
        and saving it to disk.

        Parameters
        ----------
        args
            argparse namespace containing the command line arguments passed by
            the user.
        """

        ds = Vec2VecDataset(dsname=args.dsname, debug=args.debug)

        if not args.force:
            p = ds.check_exists(directory=args.cache)
            if p is not None:
                critical = f"This RAW dataset {p} exists and override is False"
                dlog.critical(critical)
                raise RuntimeError(critical)

        ds.smart_load(directory=args.path)
        ds.init_splits(
            p_tvt=args.split, force=args.force,
            splits_override=args.override_split
        )
        ds.init_ml_data(
            scale_features=args.scale_features,
            scale_targets=args.scale_targets, force=args.force
        )
        ds.save_state(directory=args.cache, override=args.force)
