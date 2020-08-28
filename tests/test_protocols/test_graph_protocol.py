#!/usr/bin/env python3

from crescendo.datasets.qm9 import QMXDataset
from crescendo.protocols.graph_protocols import GraphToVectorProtocol
from crescendo.utils.ml_utils import seed_all


class TestGraphToVectorProtocol:

    def test(self):

        seed_all(123)

        dat = QMXDataset()
        dat.load(dummy_data=100, dummy_default_target_size=4)
        data_loaders = dat.get_data_loaders(batch_sizes=(10, 10, 80))

        protocol = GraphToVectorProtocol(
            trainLoader=data_loaders['train'],
            validLoader=data_loaders['valid']
        )
        protocol.initialize_model(
            n_node_features=dat.n_class_per_feature[0],
            n_edge_features=dat.n_class_per_feature[1],
            output_size=4
        )
        protocol.initialize_support()

        (train_loss, valid_loss, learning_rate) = protocol.train(10, clip=2)
        assert train_loss[-1] < train_loss[0]
        assert valid_loss[-1] < valid_loss[0]

        protocol.eval()
