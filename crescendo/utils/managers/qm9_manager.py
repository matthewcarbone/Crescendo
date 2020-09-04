#!/usr/bin/env python3


import os

import numpy as np
import pandas as pd
import yaml

from crescendo.utils.managers.manager_base import Manager

from crescendo.datasets.qm9 import QM9Dataset, QM9GraphDataset
from crescendo.protocols.graph_protocols import GraphToVectorProtocol
from crescendo.defaults import QM9_DS_ENV_VAR
from crescendo.utils.logger import logger_default as dlog
from crescendo.utils.ml_utils import _call_subprocess
from crescendo.utils.py_utils import check_for_environment_variable
from crescendo.utils.ml_utils import save_caches


class QM9Manager(Manager):

    def __init__(self, dsname, cache):

        if cache is None:
            cache = check_for_environment_variable(QM9_DS_ENV_VAR)

        # Location of the cache containing the datasets
        self.root_above = f"{cache}/{dsname}"
        self.dsname = dsname
        self.cache = cache

    @staticmethod
    def _try_load_all(ds, args, path):
        ds.load(path=path)
        if args.load_qm8:
            ds.load_qm8_electronic_properties(path=args.qm8_path)
        if args.load_O_xanes:
            ds.load_oxygen_xanes(path=args.O_xanes_path)
        return ds

    def init_raw(self, args):
        """Runs the initialization protocol for creating the raw dataset, and
        saving it to disk.

        Parameters
        ----------
        args
            argparse namespace containing the command line arguments passed by
            the user.
        """

        ds = QM9Dataset(dsname=args.dsname, debug=args.debug)

        if not args.force:
            p = ds.check_exists('raw', directory=args.cache)
            if p is not None:
                raise RuntimeError(
                    f"This RAW dataset {p} exists and override is False"
                )

        ds = QM9Manager._try_load_all(ds, args, path=args.qm9_path)
        ds.save_state(directory=args.cache, override=args.force)

    def init_graph(self, args):
        """Runs the initialization protocol for creating the graph dataset, and
        saving to disk.

        Parameters
        ----------
        args
            argparse namespace containing the command line arguments passed by
            the user.
        """

        ds = QM9Dataset(dsname=args.dsname)
        ds.load_state(dsname=args.dsname, directory=args.cache)

        dsG = QM9GraphDataset(ds, seed=args.seed)

        if not args.force:
            p = dsG.check_exists('mld', directory=args.cache)
            if p is not None:
                raise RuntimeError(
                    f"This MLD dataset {p} exists and override is False"
                )

        dsG.to_mol(canonical=args.canonical)
        if args.analyze:
            dsG.analyze()
        dsG.to_graph(
            node_method=args.node_method, edge_method=args.edge_method
        )
        dsG.init_ml_data(
            target_type=args.target_type,
            targets_to_use=args.targets_to_use,
            scale_targets=args.scale_targets
        )
        dsG.init_splits(p_tvt=args.split)
        dsG.save_state(directory=args.cache, override=args.force)

    def submit(self, epochs):
        """Submits jobs to the job controller."""

        # Move the script to the working directory
        script = f"{self.root_above}/submit.sh"
        _call_subprocess(f'mv {script} .')

        # Submit the jobs
        all_dirs = self._get_all_trial_dirs()
        trials = [f"{ii:03}" for ii in range(len(all_dirs))]

        for trial in trials:
            s = f'sbatch submit.sh 0 {self.dsname} {trial} {self.cache} ' \
                f'{epochs}'
            dlog.info(f"Submitting {s}")

            _call_subprocess(s)

        _call_subprocess(f'mv submit.sh {self.root_above}')

    @staticmethod
    def _eval_single_cache(cache):
        """Evaluates the results on a single cache and returns the average
        MAE on that cache. In the case of the QM9 targets, the caches are
        (index, prediction, target)."""

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

        mlds = QM9GraphDataset()
        mlds.load_state(dsname=self.dsname, directory=self.cache)

        for trial in trials:
            dlog.info(f"Evaluating trial {trial}")

            root = os.path.join(self.cache, self.dsname, trial)
            config_path = os.path.join(root, 'config.yaml')
            config = yaml.safe_load(open(config_path))
            configs[trial] = config
            data_loaders = mlds.get_loaders(config['batch_size'])

            # This will automatically load the saved checkpoint
            protocol = GraphToVectorProtocol(
                root,
                trainLoader=data_loaders['train'],
                validLoader=data_loaders['valid']
            )

            # This will automatically apply the saved checkpoint; specifically,
            # the **best** model as evaluated on the validation data.
            protocol.initialize_model(
                best=True,
                model_name='MPNN',
                n_node_features=mlds.node_edge_features[0],
                n_edge_features=mlds.node_edge_features[1],
                output_size=mlds.n_targets,
                hidden_node_size=config['hidden_node_size'],
                hidden_edge_size=config['hidden_edge_size']
            )

            # We'll always use the MAE criterion for final eval.
            protocol._init_criterion('l1')

            test_cache, valid_cache, train_cache = \
                save_caches(protocol, mlds, data_loaders)

            test_mu, test_sd = QM9Manager._eval_single_cache(test_cache)
            test_mu_list.append(test_mu)
            test_sd_list.append(test_sd)

            valid_mu, valid_sd = QM9Manager._eval_single_cache(valid_cache)
            valid_mu_list.append(valid_mu)
            valid_sd_list.append(valid_sd)

            train_mu, train_sd = QM9Manager._eval_single_cache(train_cache)
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
