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


def add_vec2vec_args(ap):
    """Adds the vector2vector parser options."""

    req = ap.add_argument_group(
        'required'
    )
    req.add_argument(
        '-d', '--dsname', dest='dsname', type=str,
        required=True, help='Sets the dataset name. Required.'
    )

    ap.add_argument(
        '-c', '--cache', dest='cache', type=str, default=None,
        help='Sets the directory to save the dataset, ML results, etc; '
        'otherwise set by the VEC2VEC_GENERAL_DS_ENV_VAR environment variable.'
    )

    subparsers = ap.add_subparsers(
        help='Execution protocols for vector-to-vector methods.',
        dest='protocol'
    )

    # Raw ---------------------------------------------------------------------
    raw_subparser = subparsers.add_parser(
        "init", formatter_class=SortingHelpFormatter,
        description='Loads in the raw data from the specified path, and '
        'converts it into machine learning-ready format as per the specified '
        'parameters in the command line interface.'
    )

    req2 = raw_subparser.add_argument_group(
        'required'
    )

    req2.add_argument(
        '--path', type=str, required=True,
        help='Sets the path that points to the directory containing the '
        'feature, target, metadata and index information. Note that we use '
        'the smart loader to do this, so the feature .csv, which should be a '
        'labeld .csv file (with headers), should contain the word "features" '
        'in its file name, same for targets, meta for metadata, and index '
        'or idx for the index information. Note also that the metadata and '
        'indexes are optional.'
    )

    raw_subparser.add_argument(
        '--split', nargs='+', type=float,
        help='Specifies the split proportions for the test, validation and '
        'training partitions.',
        default=[0.03, 0.03, 0.94]
    )
    raw_subparser.add_argument(
        '--override-split', dest='splits_override', type=str,
        help='The absolute path to a pickle file containing a pickled python '
        'dictionary with keys "train", "valid" and "test", which each contain '
        'a list of the indexes to use in generating the splits. This '
        'overrides any default options this method is to use in favor of the '
        'user-provided splits.'
    )
    raw_subparser.add_argument(
        '--scale-features', dest='scale_features', default=False,
        action='store_true',
        help='Scale feature data by the mean/sd of the training split.'
    )
    raw_subparser.add_argument(
        '--scale-targets', dest='scale_targets', default=False,
        action='store_true',
        help='Scale target data by the mean/sd of the training split.'
    )
    raw_subparser.add_argument(
        '--downsample-train', dest='downsample_train', type=int, default=None,
        help='If specified, will downsample the training set by selecting the '
        'first "downsample_train" points in the training split and using only '
        'those for training.'
    )

    # Prime ML ----------------------------------------------------------------
    prime_subparser = subparsers.add_parser(
        "prime", formatter_class=SortingHelpFormatter,
        description='Prepare for machine learning training by'
        'Creates the necessary directories and trial directories for '
        'running train in a later step.',
    )
    prime_subparser.add_argument(
        '--ml-config', dest='ml_config', type=str,
        default='configs/vec2vec_ml_config.yaml',
        help='Sets the config file path.'
    )
    prime_subparser.add_argument(
        '--slurm-config', dest='slurm_config', type=str,
        default='configs/vec2vec_slurm_config.yaml',
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

    # Evaluation
    subparsers.add_parser(
        "eval", formatter_class=SortingHelpFormatter,
        description='Evaluates all stored results and saves a summary csv '
        'file in the dataset directory. Note that once this method is called, '
        'it is recommended that no further training/evaluation be performed '
        'on that dataset, as the results from the testing set will be visible '
        'to the experimenter.'
    )


def add_qm9_args(ap):
    """Adds the QM9 parser options."""

    # Specify the dataset name
    # ap = ap.add_argument_group('General options')
    req = ap.add_argument_group(
        'required'
    )
    req.add_argument(
        '-d', '--dsname', dest='dsname', type=str,
        required=True, help='Sets the dataset name. Required.'
    )
    ap.add_argument(
        '-c', '--cache', dest='cache', type=str, default=None,
        help='Sets the directory to save the dataset, ML results, etc; '
        'otherwise set by the QM9_DS_CACHE environment variable.'
    )
    ap.add_argument(
        '-q', '--qm9-path', dest='qm9_path', type=str, default=None,
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
        'graph featurization methods'
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
        description='Prepare for machine learning training by'
        'Creates the necessary directories and trial directories for '
        'running train in a later step.',
    )
    prime_subparser.add_argument(
        '--ml-config', dest='ml_config', type=str,
        default='configs/qm9_ml_config.yaml',
        help='Sets the config file path.'
    )
    prime_subparser.add_argument(
        '--slurm-config', dest='slurm_config', type=str,
        default='configs/qm9_slurm_config.yaml',
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

    # Evaluation
    subparsers.add_parser(
        "eval", formatter_class=SortingHelpFormatter,
        description='Evaluates all stored results and saves a summary csv '
        'file in the dataset directory. Note that once this method is called, '
        'it is recommended that no further training/evaluation be performed '
        'on that dataset, as the results from the testing set will be visible '
        'to the experimenter.'
    )


def global_parser(sys_argv):
    ap = argparse.ArgumentParser(formatter_class=SortingHelpFormatter)
    ap.add_argument(
        '--debug', dest='debug', type=int, default=-1,
        help='Sets the debug flag for max number of loaded data points. '
        'If unspecified, disables debug mode.'
    )
    ap.add_argument(
        '--force', dest='force', default=False, action='store_true',
        help='Overrides failsafes for e.g. overwriting datasets.'
    )
    ap.add_argument(
        '--seed', dest='seed', type=int, default=None,
        help='Seeds the random number generators. If unspecified, will '
        'randomly seed the generators.'
    )

    subparsers = ap.add_subparsers(
        help='Global options. Each choice here represents '
        'a different project with fundamentally different datasets and likely '
        'different models as well.', dest='project'
    )

    # (1)
    qm9_subparser = subparsers.add_parser(
        "qm9", formatter_class=SortingHelpFormatter,
        description='QM9 molecular data. The general protocol here is loading '
        'SMILES and target property data from QM9 .xyz files, processing the '
        'SMILES data into graph-representations, and using a message passing '
        'neural network to make predictions on fixed length vectors. The '
        'workflow is raw -> graph -> prime -> train -> eval.'
    )
    add_qm9_args(qm9_subparser)

    # (2)
    vec2vec_subparser = subparsers.add_parser(
        "vec2vec", formatter_class=SortingHelpFormatter,
        description='Vector-to-vector data. The general protocol consists of '
        'loading the feature, target, meta and index information from '
        'individual .csv files. These must all have the same number of rows '
        'such that row `i` in any of the files corresponds to row `i` in the '
        'others. The general workflow is init -> prime -> train -> eval.'
    )
    add_vec2vec_args(vec2vec_subparser)

    return ap.parse_args(sys_argv)
