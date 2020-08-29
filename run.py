#!/usr/bin/env python3

import argparse

from crescendo.datasets.qm9 import QM9Dataset, QM9GraphDataset


def parser():
    """Parses the arguments via argparse and returns the parser.parse_args()
    object."""

    ap = argparse.ArgumentParser()

    # Dataset/machine learning core options
    ap.add_argument(
        '--dataset-raw', dest='dataset_raw', type=str, default=None,
        help='loads/inits the raw dataset and specify the dataset name'
    )
    ap.add_argument(
        '--dataset-graph', dest='dataset_graph', type=str, default=None,
        help='loads/inits the graph dataset and specify the dataset name'
    )

    # Graph options/featurization
    ap.add_argument(
        '--canonical', dest='canonical', default=False, action='store_true',
        help='if True, uses the canonical smiles instead of normal ones'
    )
    ap.add_argument(
        '--seed', dest='seed', type=int, default=None,
        help='seeds the RNG for the graph dataset'
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

    # Debugging and forcing
    ap.add_argument(
        '--debug', dest='debug', type=int, default=-1,
        # required='--prime' in sys.argv,
        help='sets the debug flag for max number of loaded data points'
    )
    ap.add_argument(
        '--override', dest='override', default=False, action='store_true',
        help='override the failsafes for overwriting datasets'
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

    args = ap.parse_args()

    return args


def try_load_all(ds, args, path):
    ds.load(path=path)
    if args.load_qm8:
        ds.load_qm8_electronic_properties(path=args.qm8_path)
    if args.load_O_xanes:
        ds.load_oxygen_xanes(path=args.O_xanes_path)
    return ds


if __name__ == '__main__':

    args = parser()

    # First we create a qm9 raw dataset
    if args.dataset_raw is not None:
        ds = QM9Dataset(dsname=args.dataset_raw, debug=args.debug)

        if not args.override:
            p = ds.check_exists('raw', directory=args.cache)
            if p is not None:
                raise RuntimeError(
                    f"This RAW dataset {p} exists and override is False"
                )

        ds = try_load_all(ds, args, path=args.dataset_raw)
        ds.save_state(directory=args.cache, override=args.override)

    # Then, we construct the graph dataset from this cached one. Note that this
    # requires the previous raw data was saved.
    elif args.dataset_graph is not None:

        ds = QM9Dataset(dsname=args.dataset_graph)
        ds.load_state(dsname=args.dataset_graph, directory=args.cache)

        dsG = QM9GraphDataset(ds, seed=args.seed)

        if not args.override:
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
        dsG.save_state(directory=args.cache, override=args.override)

    else:
        raise NotImplementedError("Unknown run protocol")
