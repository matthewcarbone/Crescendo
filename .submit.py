#!/usr/bin/env python3


"""Helper script that is not designed to be called by the user. This is called
by the code itself and is designed to be run on a compute node."""

import os
import sys
import yaml

from crescendo.scripts.on_compute import run_qm9_graph_vector, \
    run_vec2vec


if __name__ == '__main__':
    dowhat = int(sys.argv[1])
    dsname = str(sys.argv[2])
    trial = str(sys.argv[3])
    cache = str(sys.argv[4])
    epochs = int(sys.argv[5])
    config_path = os.path.join(cache, dsname, trial, 'config.yaml')
    config = yaml.safe_load(open(config_path))

    # QM9 graph -> fixed length vector
    if dowhat == 0:
        run_qm9_graph_vector(dsname, config, cache, epochs, trial=trial)

    # Fixed length -> fixed length vector standard MLP
    elif dowhat == 1:
        run_vec2vec(dsname, config, cache, epochs, trial=trial)
    else:
        raise RuntimeError("Unknown 'dowhat' protocol")
