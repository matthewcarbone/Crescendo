#!/usr/bin/env python3


"""Helper script that is not designed to be called by the user. This is called
by the code itself and is designed to be run on a compute node."""

import os
import sys
import yaml

from crescendo.scripts.qm9_graph_vector_compute import run_single_protocol


if __name__ == '__main__':
    dowhat = int(sys.argv[1])

    # QM9-to-graph predictions
    if dowhat == 0:
        dsname = str(sys.argv[2])
        trial = str(sys.argv[3])
        cache = str(sys.argv[4])
        config_path = os.path.join(cache, dsname, trial, 'config.yaml')
        config = yaml.safe_load(open(config_path))
        run_single_protocol(dsname, config, cache, trial)
    else:
        raise RuntimeError("Unknown 'dowhat' protocol")
