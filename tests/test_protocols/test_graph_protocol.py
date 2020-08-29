#!/usr/bin/env python3

from crescendo.datasets.qm9 import QM9Dataset, QM9GraphDataset
from crescendo.protocols.graph_protocols import GraphToVectorProtocol
from crescendo.utils.ml_utils import seed_all

from crescendo.defaults import QM9_TEST_DATA_PATH


class TestGraphToVectorProtocol:

    def test(self):

        seed_all(123)

        ds = QM9Dataset(dsname="TESTDS")
        ds.load(QM9_TEST_DATA_PATH)
        dsG = QM9GraphDataset(ds)
        dsG.to_mol()
        dsG.to_graph()
        dsG.init_ml_data(scale_targets=True)
        dsG.init_splits()
        data_loaders = dsG.get_loaders()

        protocol = GraphToVectorProtocol(
            trainLoader=data_loaders['train'],
            validLoader=data_loaders['valid']
        )
        protocol.initialize_model(
            n_node_features=dsG.node_edge_features[0],
            n_edge_features=dsG.node_edge_features[1],
            output_size=1
        )
        protocol.initialize_support()

        (train_loss, valid_loss, learning_rate) = protocol.train(10, clip=2)
        assert train_loss[-1] < train_loss[0]
        assert valid_loss[-1] < valid_loss[0]

        protocol.eval()
