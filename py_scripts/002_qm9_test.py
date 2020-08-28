#!/usr/bin/env python3

import pickle

from crescendo.datasets.qm9 import QMXDataset
from crescendo.protocols.graph_protocols import GraphToVectorProtocol


if __name__ == '__main__':
    ds = QMXDataset(debug=1000)
    ds.load(min_heavy_atoms=2, canonical=False)
    meta = ds.ml_ready(
        'qm9_prop',
        scale_targets=True,
        method='weave-canonical',
        target_features=[10]
    )
    data_loaders = ds.get_data_loaders(
        p_tvt=(0.05, 0.05, None),
        batch_sizes=(1000, 1000, 32),
        seed=1234
    )
    protocol = GraphToVectorProtocol(
        trainLoader=data_loaders['train'],
        validLoader=data_loaders['valid']
    )
    protocol.initialize_model(
        n_node_features=ds.n_class_per_feature[0],
        n_edge_features=ds.n_class_per_feature[1],
        output_size=1,
        hidden_node_size=64,
        hidden_edge_size=64
    )
    protocol.initialize_support(
        optimizer=('adam', {'lr': 0.5e-3}),
        scheduler=('rlrp', {'patience': 3, 'factor': 0.05, 'min_lr': 1e-6})
    )
    protocol.train(500, clip=None)

    cache = protocol.eval(meta=meta['target_metadata'])

    pickle.dump(cache, open('test_res.pkl', 'wb'), protocol=4)
