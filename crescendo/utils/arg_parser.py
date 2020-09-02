#!/usr/bin/env python3

import argparse
from argparse import HelpFormatter, ArgumentDefaultsHelpFormatter
from operator import attrgetter
import sys


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
        help='Sets the dataset name and is required.'
    )
    ap.add_argument(
        '--cache', dest='cache', type=str, default=None,
        help='Sets the directory to save the dataset, ML results, etc; '
        'otherwise set by the QM9_DS_CACHE environment variable.'
    )
    ap.add_argument(
        '--qm9-path', dest='qm9_path', type=str, default=None,
        help='Sets the path for various datasets; otherwise set by the '
        'QM9_DATA_PATH environment variable.'
    )

    subparsers = ap.add_subparsers(
        help='Execution protocols for qm9.', dest='protocol'
    )

    # Raw ---------------------------------------------------------------------
    raw_subparser = subparsers.add_parser(
        "raw", formatter_class=SortingHelpFormatter,
        description='Loads in the raw data from various sources and properly '
        'pairs them together in a concise way.'
    )
    raw_subparser.add_argument(
        '--qm8-path', dest='qm8_path', type=str, default=None,
        help='Sets the QM8 path; otherwise set by the QM8_EP_DATA_PATH '
        'environment variable.'
    )
    raw_subparser.add_argument(
        '--oxygen-xanes-path', dest='O_xanes_path', type=str, default=None,
        help='Sets the Oxygen XANES path; otherwise set by the '
        'QM9_O_FEFF_PATH environment variable.'
    )
    raw_subparser.add_argument(
        '--no-qm8', dest='load_qm8', default=True, action='store_false',
        help='Switch off the default loading of QM8 electronic properties.'
    )
    raw_subparser.add_argument(
        '--no-oxygen-xanes', dest='load_O_xanes', default=True,
        action='store_false', help='Switch off the default loading of '
        'Oxygen XANES spectra.'
    )

    # Graph -------------------------------------------------------------------
    graph_subparser = subparsers.add_parser(
        "graph", formatter_class=SortingHelpFormatter,
        description='Initializes the graph objects and the QM9GraphDataset, '
        'and prepares these objects for machine learning training later.'
    )

    graph_subparser.add_argument(
        '--analyze', dest='analyze', default=False, action='store_true',
        help='Runs the analysis method on the graphs, initializing the '
        '"summary" attribute for each QM9DataPoint; useful for performing '
        'analysis afterwards via e.g. loading the dataset into a Jupyter '
        'notebook. Note that this can take a while!'
    )
    graph_subparser.add_argument(
        '--split', nargs='+', type=float,
        help='Specifies the split proportions for the test, validation and '
        'training partitions.',
        default=[0.03, 0.03, 0.94]
    )

    methods = graph_subparser.add_argument_group(
        'Graph featurization methods.'
    )
    methods.add_argument(
        '--node-method', dest='node_method', type=str, default='weave',
        help='Sets the node featurization method.'
    )
    methods.add_argument(
        '--edge-method', dest='edge_method', type=str, default='canonical',
        help='Sets the edge featurization method.'
    )
    methods.add_argument(
        '--canonical', dest='canonical', default=False, action='store_true',
        help='Uses the canonical smiles instead of normal ones.'
    )

    targets = graph_subparser.add_argument_group(
        'target options'
    )
    targets.add_argument(
        '--target-type', dest='target_type', type=str, default='qm9properties',
        help='Sets the target type.',
        choices=['qm9properties', 'qm8properties', 'oxygenXANES']
    )
    targets.add_argument(
        '--targets-to-use', dest='targets_to_use', nargs='+', type=int,
        help='Specifies the indexes of the targets to use.',
        default=[10]
    )
    targets.add_argument(
        '--scale-targets', dest='scale_targets', default=False,
        action='store_true',
        help='Scales the targets to zero mean and unit variance.'
    )

    # Prime ML ----------------------------------------------------------------
    prime_subparser = subparsers.add_parser(
        "prime", formatter_class=SortingHelpFormatter,
        description='prepare for machine learning training by'
        'Creates the necessary directories and trial directories for '
        'running train in a later step.',
    )
    prime_subparser.add_argument(
        '--ml-config', dest='ml_config', type=str,
        default='configs/ml_config_template.yaml',
        help='Sets the config file path.'
    )
    prime_subparser.add_argument(
        '--slurm-config', dest='slurm_config', type=str,
        default='configs/slurm_config_template.yaml',
        help='Sets the SLURM config file path.'
    )
    prime_subparser.add_argument(
        '--max-hp', dest='max_hp', type=int, default=24,
        help='Maximum number of hyper parameters to use in one shot.'
    )

    # Training ----------------------------------------------------------------
    execute_subparser = subparsers.add_parser(
        "train", formatter_class=SortingHelpFormatter,
        description='Runs ML training via submitting to the SLURM job '
        'controller.'
    )
    execute_subparser.add_argument(
        '--epochs', dest='epochs', type=int, required=True,
        help='Specifies the number of epochs to train for; required.'
    )


def global_parser(sys_argv):
    ap = argparse.ArgumentParser(formatter_class=SortingHelpFormatter)
    ap.add_argument(
        '--debug', dest='debug', type=int, default=-1,
        help='Sets the debug flag for max number of loaded data points.'
    )
    ap.add_argument(
        '--force', dest='force', default=False, action='store_true',
        help='Overrides failsafes for e.g. overwriting datasets.'
    )
    ap.add_argument(
        '--seed', dest='seed', type=int, default=None,
        help='Seeds the random number generators.'
    )

    # Initialize qm9 subparser
    subparsers = ap.add_subparsers(
        help='Global options. Each option here represents '
        'a different project with fundamentally different datasets and likely '
        'different models as well', dest='project'
    )
    qm9_subparser = subparsers.add_parser(
        "qm9", formatter_class=SortingHelpFormatter
    )
    # matrix_subparser = subparsers.add_parser("--matrix")
    add_qm9_args(qm9_subparser)

    return ap.parse_args(sys_argv)
