#!/usr/bin/env python3

import argparse
from argparse import HelpFormatter, ArgumentDefaultsHelpFormatter
from operator import attrgetter


# https://stackoverflow.com/questions/
# 12268602/sort-argparse-help-alphabetically
class SortingHelpFormatter(ArgumentDefaultsHelpFormatter, HelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)
        # self._max_help_position = 40
        # self._width = 100


def add_qm9_args(ap):
    """Adds the QM9 parser options."""

    # Specify the dataset name
    # ap = ap.add_argument_group('General options')
    ap.add_argument(
        '-d', '--dsname', dest='dsname', type=str, required=True,
        help='set the dataset name; required'
    )
    ap.add_argument(
        '--cache', dest='cache', type=str, default=None,
        help='sets the directory to save the dataset, ML results, etc; '
        'otherwise set by the QM9_DS_CACHE environment variable'
    )
    ap.add_argument(
        '--qm9-path', dest='qm9_path', type=str, default=None,
        help='set the path for various datasets; otherwise set by the '
        'QM9_DATA_PATH environment variable'
    )

    subparsers = ap.add_subparsers(
        help='execution protocols for qm9'
    )

    # Raw ---------------------------------------------------------------------
    raw_subparser = subparsers.add_parser(
        "raw", formatter_class=SortingHelpFormatter
    )
    raw_subparser.add_argument(
        '--qm8-path', dest='qm8_path', type=str, default=None,
        help='sets the QM8 path'
    )
    raw_subparser.add_argument(
        '--oxygen-xanes-path', dest='O_xanes_path', type=str, default=None,
        help='sets the Oxygen XANES path'
    )
    raw_subparser.add_argument(
        '--no-qm8', dest='load_qm8', default=True, action='store_false',
        help='no loading QM8 electronic properties'
    )
    raw_subparser.add_argument(
        '--no-oxygen-xanes', dest='load_O_xanes', default=True,
        action='store_false', help='no loading Oxygen XANES spectra'
    )

    # Graph -------------------------------------------------------------------
    graph_subparser = subparsers.add_parser(
        "graph", formatter_class=SortingHelpFormatter
    )

    graph_subparser.add_argument(
        '--analyze', dest='analyze', default=False, action='store_true',
        help='runs the analysis method on the graphs, initializing the '
        '"summary" attribute for each QM9DataPoint; useful for performing '
        'analysis afterwards via e.g. loading the dataset into a Jupyter '
        'notebook'
    )
    graph_subparser.add_argument(
        'split', nargs='+', type=float,
        help='specify the split proportions for test validation and train'
    )

    methods = graph_subparser.add_argument_group(
        'graph featurization methods'
    )
    methods.add_argument(
        '--node-method', dest='node_method', type=str, default='weave',
        help='sets the node featurization method'
    )
    methods.add_argument(
        '--edge-method', dest='edge_method', type=str, default='canonical',
        help='sets the edge featurization method'
    )
    methods.add_argument(
        '--canonical', dest='canonical', default=False, action='store_true',
        help='uses the canonical smiles instead of normal ones'
    )

    targets = graph_subparser.add_argument_group(
        'target options'
    )
    targets.add_argument(
        '--target-type', dest='target_type', type=str, default='qm9properties',
        help='sets the target type',
        choices=['qm9properties', 'qm8properties', 'oxygenXANES']
    )
    targets.add_argument(
        '--targets-to-use', dest='targets_to_use', nargs='+', type=int,
        help='specify the indexes of the targets to use',
        default=[10]
    )
    targets.add_argument(
        '--scale-targets', dest='scale_targets', default=False,
        action='store_true',
        help='if True, scales the targets to zero mean and unit variance'
    )

    # Machine learning --------------------------------------------------------
    prime_subparser = subparsers.add_parser(
        "ml", formatter_class=SortingHelpFormatter
    )
    prime_groups = prime_subparser.add_argument_group(
        'Prepare machine learning training'
    )
    prime_groups.add_argument(
        '--prime', dest='to_prime', default=False, action='store_true',
        help='creates the necessary directories and trial directories for '
        'running --train in a later step'
    )
    prime_groups.add_argument(
        '--ml-config', dest='ml_config', type=str,
        default='configs/ml_config_template.yaml',
        help='sets the config file path'
    )
    prime_groups.add_argument(
        '--slurm-config', dest='slurm_config', type=str,
        default='configs/slurm_config.yaml',
        help='sets the SLURM config file path'
    )
    prime_groups.add_argument(
        '--max-hp', dest='max_hp', type=int, default=24,
        help='maximum number of hyperparameters to use in one shot'
    )

    execute_groups = prime_subparser.add_argument_group(
        'Run machine learning training'
    )
    execute_groups.add_argument(
        '--train', dest='to_train', default=False, action='store_true',
        help='runs ML training via submitting to the SLURM job controller'
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
