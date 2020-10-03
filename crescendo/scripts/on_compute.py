#!/usr/bin/env python3

import os
import uuid


from crescendo.utils.ml_utils import save_caches


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

    from crescendo.datasets.qm9 import QM9GraphDataset
    from crescendo.protocols.graph_protocols import GraphToVectorProtocol

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

    # Note that it is highly likely the training process does not reach this
    # final state. However, the train method has internal checkpointing
    # capability, so the model can always be reloaded.
    save_caches(protocol, mlds, data_loaders)


def run_vec2vec(
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

    from crescendo.datasets.vec2vec import Vec2VecDataset
    from crescendo.protocols.vec2vec_protocols import Vec2VecProtocol

    mlds = Vec2VecDataset(dsname=dsname)
    mlds.load_state(directory=cache)
    data_loaders = mlds.get_loaders(config['batch_size'])

    root = os.path.join(cache, dsname, trial)
    protocol = Vec2VecProtocol(
        root,
        trainLoader=data_loaders['train'],
        validLoader=data_loaders['valid'],
        parallel=True
    )

    protocol.initialize_model(
        mlds.n_features, mlds.n_targets, mlds.n_meta,
        model_type=config['model_type'],
        input_size=mlds.n_features,
        hidden_size=config['hidden_size'],
        output_size=mlds.n_targets,
        n_hidden_layers=config['n_hidden_layers'],
        dropout=config['dropout']
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

    # Note that it is highly likely the training process does not reach this
    # final state. However, the train method has internal checkpointing
    # capability, so the model can always be reloaded.
    save_caches(protocol, mlds, data_loaders)
