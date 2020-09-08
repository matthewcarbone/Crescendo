#!/usr/bin/env python3


import os

import numpy as np
import pandas as pd
import pickle
import yaml

from crescendo.utils.managers.manager_base import Manager

from crescendo.datasets.vec2vec import Vec2VecDataset
from crescendo.protocols.vec2vec_protocols import Vec2VecProtocol
from crescendo.defaults import VEC2VEC_GENERAL_DS_ENV_VAR
from crescendo.utils.logger import logger_default as dlog
from crescendo.utils.py_utils import check_for_environment_variable
from crescendo.utils.ml_utils import save_caches, _call_subprocess


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

        override = None
        if args.splits_override is not None:
            override = pickle.load(open(args.splits_override, 'rb'))

        ds.init_splits(
            p_tvt=args.split, force=args.force, splits_override=override,
            downsample_train=args.downsample_train
        )
        ds.init_ml_data(
            scale_features=args.scale_features,
            scale_targets=args.scale_targets, force=args.force
        )
        ds.save_state(directory=args.cache, override=args.force)

    def submit(self, epochs):
        """Submits jobs to the job controller."""

        # Move the script to the working directory
        script = f"{self.root_above}/submit.sh"
        _call_subprocess(f'mv {script} .')

        # Submit the jobs
        all_dirs = self._get_all_trial_dirs()
        trials = [f"{ii:03}" for ii in range(len(all_dirs))]

        for trial in trials:
            s = f'sbatch submit.sh 1 {self.dsname} {trial} {self.cache} ' \
                f'{epochs}'
            dlog.info(f"Submitting {s}")

            _call_subprocess(s)

        _call_subprocess(f'mv submit.sh {self.root_above}')

    @staticmethod
    def _eval_single_cache(cache):
        maes = [np.mean(np.abs(xx[1] - xx[2])) for xx in cache]
        return np.mean(maes), np.std(maes)

    def eval(self, force):
        """After training on various hyperparameter sets, the data should be
        evalated on the testing set. Note that after evaluation, there should
        be no more fine-tuning of the network, as once the testing set is
        viewed, any further changes to the network and any re-evaluation will
        introduce human bias into the results. Thus, once this method is
        called, a FINAL_SUMMARY.csv text file will be saved in the dataset
        directory as a reminder that eval cannot be called again. Of course,
        the user can simply cheat this failsafe with --force or by simply
        deleting FINAL_SUMMARY.csv nevertheless this serves as a reminder to
        use good practice.

        Parameters
        ----------
        force : bool
            If True, and the summary already exists, this script will log a
            warning, and overwrite the old summary. Default is False.
        """

        summary_path = f"{self.root_above}/FINAL_SUMMARY.csv"
        if os.path.exists(summary_path) and force:
            dlog.warning(
                "FINAL_SUMMARY exists for this dataset and force is True. "
                "Note that you should be aware that if you are still "
                "fine-tuning the network, you could be introducing human bias "
                "into the results. Please ensure this is intended behavior."
            )
        elif os.path.exists(summary_path) and not force:
            dlog.error(
                "FINAL_SUMMARY exists for this dataset and force is False. "
                "Exiting without re-evaluating."
            )
            return

        all_dirs = self._get_all_trial_dirs()
        trials = [f"{ii:03}" for ii in range(len(all_dirs))]
        train_mu_list = []
        train_sd_list = []
        valid_mu_list = []
        valid_sd_list = []
        test_mu_list = []
        test_sd_list = []
        configs = dict()

        mlds = Vec2VecDataset(dsname=self.dsname)
        mlds.load_state(directory=self.cache)

        for trial in trials:
            dlog.info(f"Evaluating trial {trial}")

            root = os.path.join(self.cache, self.dsname, trial)
            config_path = os.path.join(root, 'config.yaml')
            config = yaml.safe_load(open(config_path))
            configs[trial] = config
            data_loaders = mlds.get_loaders(config['batch_size'])

            # This will automatically load the saved checkpoint
            protocol = Vec2VecProtocol(
                root,
                trainLoader=data_loaders['train'],
                validLoader=data_loaders['valid']
            )

            # This will automatically apply the saved checkpoint; specifically,
            # the **best** model as evaluated on the validation data.
            protocol.initialize_model(
                best=True,
                model_type=config['model_type'],
                input_size=mlds.n_features,
                hidden_size=config['hidden_size'],
                output_size=mlds.n_targets,
                n_hidden_layers=config['n_hidden_layers'],
                dropout=config['dropout']
            )

            # We'll always use the MAE criterion for final eval.
            protocol._init_criterion('l1')

            test_cache, valid_cache, train_cache = \
                save_caches(protocol, mlds, data_loaders)

            test_mu, test_sd = Vec2VecManager._eval_single_cache(test_cache)
            test_mu_list.append(test_mu)
            test_sd_list.append(test_sd)

            valid_mu, valid_sd = Vec2VecManager._eval_single_cache(valid_cache)
            valid_mu_list.append(valid_mu)
            valid_sd_list.append(valid_sd)

            train_mu, train_sd = Vec2VecManager._eval_single_cache(train_cache)
            train_mu_list.append(train_mu)
            train_sd_list.append(train_sd)

            dlog.info(f"\t train {train_mu:.04e} +/- {train_sd:.04e}")
            dlog.info(f"\t valid {valid_mu:.04e} +/- {valid_sd:.04e}")
            dlog.info(f"\t test  {test_mu:.04e} +/- {test_sd:.04e}")

        results_df = pd.DataFrame({
            'trial': trials, 'train_MAE': train_mu_list,
            'train_SD': train_sd_list, 'valid_MAE': valid_mu_list,
            'valid_SD': valid_sd_list, 'test_MAE': test_mu_list,
            'test_SD': test_sd_list
        })
        results_df.sort_values(by='valid_MAE', inplace=True)
        results_df.to_csv(summary_path)
        dlog.info(f"Done: saved to {summary_path}")

        best_trial = results_df.iloc[0, 0]
        dlog.info(f"Best trial is {best_trial} with configuration")
        for key, value in configs[best_trial].items():
            dlog.info(f"\t {key} - {value}")
