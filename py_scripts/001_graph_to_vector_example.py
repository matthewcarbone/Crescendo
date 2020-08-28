#!/usr/bin/env python3

from crescendo.datasets.qm9 import QMXDataset
from crescendo.protocols.graph_protocols import GraphToVectorProtocol


if __name__ == '__main__':
    qm9_dat = QMXDataset()
    qm9_dat.load(dummy_data=1000, dummy_default_target_size=4)
    data_loaders = qm9_dat.get_data_loaders(batch_sizes=(100, 100, 800))
    protocol = GraphToVectorProtocol(
        trainLoader=data_loaders['train'],
        validLoader=data_loaders['valid']
    )
    protocol.initialize_model(
        n_node_features=qm9_dat.n_class_per_feature[0],
        n_edge_features=qm9_dat.n_class_per_feature[1],
        output_size=4
    )
    protocol.initialize_support()
    protocol.train(10, clip=2)
