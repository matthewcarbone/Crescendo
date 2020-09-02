#!/usr/bin/env python3

import sys

from crescendo.utils.arg_parser import global_parser


if __name__ == '__main__':
    args = global_parser(sys.argv[1:])

    if args.project == 'qm9':

        from crescendo.utils.managers import QM9Manager
        manager = QM9Manager(dsname=args.dsname, cache=args.cache)

        # First we create a qm9 raw dataset
        if args.protocol == 'raw':
            manager.init_raw(args)

        # Then, we construct the graph dataset from this cached one. Note that
        # this requires the previous raw data was saved.
        elif args.protocol == 'graph':
            manager.init_graph(args)

        # Load in the Graph Dataset and the ml_config.yaml file (which should
        # be) in the working directory, and execute training.
        elif args.protocol == 'prime':
            manager.prime(config_path=args.ml_config, max_hp=args.max_hp)
            manager.write_SLURM_script(slurm_config=args.slurm_config)

        # Run training
        elif args.protocol == 'train':
            manager.submit()

        else:
            raise NotImplementedError("Unknown run protocol - qm9")

    else:
        raise NotImplementedError(f"Unknown run protocol")
