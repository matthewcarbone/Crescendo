#!/usr/bin/env python3

import argparse
from argparse import HelpFormatter
from operator import attrgetter


# https://stackoverflow.com/questions/
# 12268602/sort-argparse-help-alphabetically
class SortingHelpFormatter(HelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)
        self._max_help_position = 30
        self._width = 150


def add_qm9_args(ap):
    """Adds the QM9 parser options."""

    # parent_parser = argparse.ArgumentParser(add_help=False)

    # parent_parser.add_argument(
    #     '--init', dest='do_what', type=str, default=None,
    #     help='loads/inits the raw dataset and specify the dataset name'
    # )

    # Dataset/machine learning core options
    ap.add_argument(
        '--dataset-raw', dest='dataset_raw', type=str, default=None,
        help='loads/inits the raw dataset and specify the dataset name'
    )
    ap.add_argument(
        '--dataset-graph', dest='dataset_graph', type=str, default=None,
        help='loads/inits the graph dataset and specify the dataset name'
    )
    ap.add_argument(
        '--train-prime', dest='train_prime', type=str, default=None,
        help='primes for ML training and specify the dataset name'
    )
    ap.add_argument(
        '--train-run', dest='train_run', type=str, default=None,
        help='runs ML training and specify the dataset name'
    )

    # Run ML options
    ap.add_argument(
        '--config', dest='config', type=str, default='config.yaml',
        help='sets the config file path'
    )
    ap.add_argument(
        '--slurm-config', dest='slurm_config', type=str,
        default='slurm_config.yaml',
        help='sets the SLURM config file path'
    )
    ap.add_argument(
        '--max-hp', dest='max_hp', type=int, default=24,
        help='maximum number of hyperparameters to use in one shot'
    )

    # Graph options/featurization
    ap.add_argument(
        '--canonical', dest='canonical', default=False, action='store_true',
        help='if True, uses the canonical smiles instead of normal ones'
    )
    ap.add_argument(
        '--analyze', dest='analyze', default=False, action='store_true',
        help='if True, runs the analysis method on the graphs'
    )
    ap.add_argument(
        '--node-method', dest='node_method', type=str, default='weave',
        help='sets the node featurization method'
    )
    ap.add_argument(
        '--edge-method', dest='edge_method', type=str, default='canonical',
        help='sets the edge featurization method'
    )

    # Init ML options
    ap.add_argument(
        '--target-type', dest='target_type', type=str, default='qm9properties',
        help='sets the target type',
        choices=['qm9properties', 'qm8properties', 'oxygenXANES']
    )
    ap.add_argument(
        '--targets-to-use', dest='targets_to_use', nargs='+', type=int,
        help='specify the indexes of the targets to use',
        default=[10]
    )
    ap.add_argument(
        '--scale-targets', dest='scale_targets', default=False,
        action='store_true',
        help='if True, scales the targets to zero mean and unit variance'
    )

    # Init split options
    ap.add_argument(
        '--split', dest='split', nargs='+', type=float,
        help='specify the split proportions for test validation and train',
        default=[0.05, 0.05, None]
    )

    # Paths
    ap.add_argument(
        '--qm8-path', dest='qm8_path', type=str, default=None,
        help='sets the QM8 path'
    )
    ap.add_argument(
        '--oxygen-xanes-path', dest='O_xanes_path', type=str, default=None,
        help='sets the Oxygen XANES path'
    )
    ap.add_argument(
        '--cache-location', dest='cache', type=str, default=None,
        help='sets the directory to save the dataset, ML results, etc'
    )

    # Options to skip loading various raw data. For example, if you do not have
    # access to the XANES data, you should use the flag --no-oxygen-xanes
    ap.add_argument(
        '--no-qm8', dest='load_qm8', default=True, action='store_false',
        help='no loading QM8 electronic properties'
    )
    ap.add_argument(
        '--no-oxygen-xanes', dest='load_O_xanes', default=True,
        action='store_false', help='no loading Oxygen XANES spectra'
    )


def global_parser():
    ap = argparse.ArgumentParser(formatter_class=SortingHelpFormatter)
    ap.add_argument(
        '--debug', dest='debug', type=int, default=-1,
        help='sets the debug flag for max number of loaded data points'
    )
    ap.add_argument(
        '--force', dest='force', default=False, action='store_true',
        help='override failsafes for e.g. overwriting datasets'
    )
    ap.add_argument(
        '--seed', dest='seed', type=int, default=None,
        help='seeds the RNG'
    )
    ap.add_argument(
        '--path', dest='path', type=str, default=None,
        help='set the path for various datasets'
    )

    # Initialize qm9 subparser
    subparsers = ap.add_subparsers(
        help='overall options for the project'
    )
    qm9_subparser = subparsers.add_parser(
        "qm9", formatter_class=SortingHelpFormatter
    )
    # matrix_subparser = subparsers.add_parser("--matrix")
    add_qm9_args(qm9_subparser)

    return ap.parse_args()
