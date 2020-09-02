#!/usr/bin/env python3


import os

import random

import yaml

from crescendo.datasets.qm9 import QM9Dataset, QM9GraphDataset
from crescendo.defaults import QM9_DS_ENV_VAR
from crescendo.utils.logger import logger_default as dlog
from crescendo.utils.ml_utils import read_config, \
    execution_parameters_permutations, _call_subprocess
from crescendo.utils.py_utils import check_for_environment_variable


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

        ds = QM9Dataset(dsname=args.dataset_raw, debug=args.debug)

        if not args.force:
            p = ds.check_exists('raw', directory=args.cache)
            if p is not None:
                raise RuntimeError(
                    f"This RAW dataset {p} exists and override is False"
                )

        ds = QM9Manager.try_load_all(ds, args, path=args.qm9_path)
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

        ds = QM9Dataset(dsname=args.dataset_graph)
        ds.load_state(dsname=args.dataset_graph, directory=args.cache)

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