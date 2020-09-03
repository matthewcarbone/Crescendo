#!/usr/bin/env python3


import os
import random

import numpy as np
import pandas as pd
import yaml

from crescendo.datasets.qm9 import QM9Dataset, QM9GraphDataset
from crescendo.protocols.graph_protocols import GraphToVectorProtocol
from crescendo.defaults import QM9_DS_ENV_VAR
from crescendo.utils.logger import logger_default as dlog
from crescendo.utils.ml_utils import read_config, \
    execution_parameters_permutations, _call_subprocess
from crescendo.utils.py_utils import check_for_environment_variable
from crescendo.utils.ml_utils import save_caches


class Manager:
    """Helps keep track of the different trials being performed, and writes
    important information to disk.

    Parameters
    ----------
    dsname : str
        The name of the dataset which must match that of the previously loaded
        and configured datasets.
    directory : str
        The location of the cache directory
    """

    def prime(self, config_path='configs/ml_config.yaml', max_hp=24):
        """Primes the computation by creating the necessary trial directories
        and parsing the configurations into the right places. This is
        essentially the setup to hyperparameter tuning, if desired, or running
        many different combinations of hyperparameters.

        Parameters
        ----------
        config_path : str
            The absolute path to the configuration file used to setup the
            trials.
        max_hp : int
            The maximum number of trials to generate at once. If the number of
            total hyperparameter combinations is more than max_hp, they will
            be selected randomly from the permutations.
        """

        dlog.info("Priming machine learning combinations")
        config = read_config(config_path)
        combinations = execution_parameters_permutations(config)
        dlog.info(f"Total of {len(combinations)} created")
        if len(combinations) > max_hp:
            combinations = random.sample(combinations, max_hp)
            dlog.warning(
                f"Length of all combinations exceeds max_hp of {max_hp} - "
                f"new length is {len(combinations)}"
            )

        cc = 0
        for combo in combinations:
            d = f"{self.root_above}/{cc:03}"
            os.makedirs(d)
            path = f"{d}/config.yaml"
            with open(path, 'w') as f:
                yaml.dump(combo, f, default_flow_style=False)
            dlog.info(f"Trial {cc:03} saved at {path}")
            cc += 1

    def _get_all_trial_dirs(self):
        all_dirs = os.listdir(self.root_above)
        all_dirs = [os.path.join(self.root_above, d) for d in all_dirs]
        return [xx for xx in all_dirs if os.path.isdir(xx)]

    def write_SLURM_script(self, slurm_config='configs/slurm_config.yaml'):
        """Writes the SLURM submission script to the root directory by
        detecting the number of jobs to submit."""

        # Get the directories in the root:
        all_dirs = self._get_all_trial_dirs()

        dlog.info(f"Detected {len(all_dirs)} trial directories")

        # Load the config
        dlog.info(f"Loading SLURM config from {slurm_config}")
        slurm_config = yaml.safe_load(open(slurm_config))

        with open(f"{self.root_above}/submit.sh", 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("\n")
            f.write(f"#SBATCH --job-name=c-{self.dsname} \n")
            f.write(f"#SBATCH -p {slurm_config['partition']}\n")
            f.write(f"#SBATCH -t {slurm_config['runtime']}\n")
            f.write(f"#SBATCH --account={slurm_config['account']}\n")
            f.write(f"#SBATCH -N {slurm_config['nodes']}\n")
            f.write(f"#SBATCH -n {slurm_config['tasks']}\n")
            if slurm_config['constraint'] is not None:
                f.write(f"#SBATCH -C {slurm_config['constraint']}\n")
            f.write(f"#SBATCH --qos {slurm_config['qos']}\n")
            if slurm_config['ngpu'] > 0:
                f.write(f"#SBATCH --gres=gpu:{slurm_config['ngpu']}\n")
            if slurm_config['exclusive']:
                f.write("#SBATCH --exclusive\n")
            f.write(f"#SBATCH --output=job_data/{self.dsname}_%A.out\n")
            f.write(f"#SBATCH --error=job_data/{self.dsname}_%A.err\n")
            f.write("\n")

            if slurm_config['ngpu'] > 0:
                gpuidx = [str(ii) for ii in range(slurm_config['ngpu'])]
                gpuidx_str = ','.join(gpuidx)
                f.write(f"export CUDA_VISIBLE_DEVICES={gpuidx_str}\n")
            else:
                f.write("export CUDA_VISIBLE_DEVICES=\n")
            f.write('\n')

            f.write('python3 .submit.py "$@"\n')

        dlog.info(f"Wrote SLURM script to {self.root_above}/submit.sh")


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
        results_df.sort_values(by='test_MAE', inplace=True)
        results_df.to_csv(summary_path)
        dlog.info(f"Done: saved to {summary_path}")

        best_trial = results_df.iloc[0, 0]
        dlog.info(f"Best trial is {best_trial} with configuration")
        for key, value in configs[best_trial].items():
            dlog.info(f"\t {key} - {value}")
