#!/usr/bin/env python3

import pickle

from crescendo.datasets.qm9 import QMXDataset
from crescendo.protocols.graph_protocols import GraphToVectorProtocol


if __name__ == '__main__':
    ds = QMXDataset()
    ds.load(min_heavy_atoms=2, canonical=False)
    meta = ds.ml_ready(
        'qm9_prop',
        scale_targets=True,
        atom_feature_list=['type', 'hybridization'],
        bond_feature_list=['type'],
        target_features=[0]
    )
    data_loaders = ds.get_data_loaders(
        p_tvt=(0.05, 0.05, None),
        batch_sizes=(1000, 1000, 3200)
    )
    protocol = GraphToVectorProtocol(
        trainLoader=data_loaders['train'],
        validLoader=data_loaders['valid']
    )
    protocol.initialize_model(
        n_node_features=ds.n_class_per_feature[0],
        n_edge_features=ds.n_class_per_feature[1],
        output_size=1
    )
    protocol.initialize_support()
    protocol.train(500, clip=2)

    cache = protocol.eval(meta=meta['target_metadata'])

    pickle.dump(cache, open('test_res.pkl', 'wb'), protocol=4)
