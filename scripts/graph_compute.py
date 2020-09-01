#!/usr/bin/env python3

import os
import sys
import uuid
import yaml


def run_single_protocol(dsname, config, cache, trial=str(uuid.uuid4())):
    """Initializes a machine learning protocol from a dictionary of
    parameters.

    Parameters
    ----------
    config : dict
        Must have a 1-to-1 correspondence between keys and ML parameters.
    dsname : str
    trial : str
        Defaults to a random hash if unspecified.
    """

    # Change directory one level up
    os.chdir("..")
    from crescendo.datasets.qm9 import QM9GraphDataset
    from crescendo.protocols.graph_protocols import GraphToVectorProtocol
    from crescendo.utils.training_utils import save_caches

    mlds = QM9GraphDataset(dsname)
    mlds.load_state(dsname=dsname, directory=cache)
    data_loaders = mlds.get_loaders()

    protocol = GraphToVectorProtocol(
        dsname, trial,
        trainLoader=data_loaders['train'],
        validLoader=data_loaders['valid']
    )

    protocol.initialize_model(
        n_node_features=mlds.node_edge_features[0],
        n_edge_features=mlds.node_edge_features[1],
        output_size=mlds.n_targets,
        hidden_node_size=config['hidden_node_size'],
        hidden_edge_size=config['hidden_edge_size']
    )

    protocol.initialize_support(
        optimizer=(
            config['optimizer'], {
                'lr': config['lr']
            }
        ),
        scheduler=(
            'rlrp', {
                'patience': config['patience'],
                'factor': config['factor'],
                'min_lr': config['min_lr']
            }
        )
    )

    protocol.train(config['epochs'], clip=config['clip'])
    save_caches(protocol, mlds, data_loaders)


if __name__ == '__main__':
    dsname = str(sys.argv[1])
    root = str(sys.argv[2])
    trial_str = str(sys.argv[3])
    cache = str(sys.argv[4])
    config_path = os.path.join(root, trial_str, cache, 'config.yaml')
    config = yaml.safe_load(open(config_path))
    run_single_protocol(dsname, config, trial_str)
