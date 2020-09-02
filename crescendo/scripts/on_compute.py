#!/usr/bin/env python3

import os
import uuid


from crescendo.datasets.qm9 import QM9GraphDataset
from crescendo.protocols.graph_protocols import GraphToVectorProtocol
from crescendo.utils.training_utils import save_caches


def run_qm9_graph_vector(
    dsname, config, cache, epochs, trial=str(uuid.uuid4())
):
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

    mlds = QM9GraphDataset()
    mlds.load_state(dsname=dsname, directory=cache)
    data_loaders = mlds.get_loaders(config['batch_size'])

    root = os.path.join(cache, dsname, trial)
    protocol = GraphToVectorProtocol(
        root,
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

    protocol.train(epochs, clip=config['clip'])
    save_caches(protocol, mlds, data_loaders)