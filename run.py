#!/usr/bin/env python3

import sys

from crescendo.utils.arg_parser import global_parser


def try_load_all(ds, args, path):
    ds.load(path=path)
    if args.load_qm8:
        ds.load_qm8_electronic_properties(path=args.qm8_path)
    if args.load_O_xanes:
        ds.load_oxygen_xanes(path=args.O_xanes_path)
    return ds


if __name__ == '__main__':

    args = global_parser()

    if 'qm9' in sys.argv:

        # First we create a qm9 raw dataset
        if args.dataset_raw is not None:

            # Importing here allows us to run other methods that don't require
            # these packages, without importing them.
            from crescendo.datasets.qm9 import QM9Dataset, QM9GraphDataset

            ds = QM9Dataset(dsname=args.dataset_raw, debug=args.debug)

            if not args.force:
                p = ds.check_exists('raw', directory=args.cache)
                if p is not None:
                    raise RuntimeError(
                        f"This RAW dataset {p} exists and override is False"
                    )

            ds = try_load_all(ds, args, path=args.path)
            ds.save_state(directory=args.cache, override=args.force)

        # Then, we construct the graph dataset from this cached one. Note that
        # this requires the previous raw data was saved.
        elif args.dataset_graph is not None:

            from crescendo.datasets.qm9 import QM9Dataset, QM9GraphDataset

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

        # Load in the Graph Dataset and the ml_config.json file (which should
        # be) in the working directory, and execute training.
        elif args.train_prime is not None:
            from crescendo.utils.training_utils import QM9Manager
            manager = QM9Manager(args.train_prime, directory=args.cache)
            manager.prime(config_path=args.config, max_hp=args.max_hp)
            manager.write_SLURM_script(slurm_config=args.slurm_config)

        elif args.train_run is not None:
            from crescendo.utils.training_utils import QM9Manager
            manager = QM9Manager(args.train_run, directory=args.cache)
            manager.submit()

        else:
            raise NotImplementedError("Unknown run protocol - qm9")

    else:
        raise NotImplementedError(f"Unknown run protocol")
