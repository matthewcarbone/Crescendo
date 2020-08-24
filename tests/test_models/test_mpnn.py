#!/usr/bin/env python3

from crescendo.models.mpnn import MPNN
from crescendo.utils.graphs import graph_to_vector_dummy_dataset


class TestMPNN:

    def test(self):
        """Extremely basic test that asserts that forward prop works."""

        dat = graph_to_vector_dummy_dataset(
            100, graph_max_size=10, graph_max_n_class=7,
            graph_max_e_class=5, target_size=4
        )
        mpnn = MPNN(
            n_node_features=[7],
            n_edge_features=[5],
            output_size=4
        )
        for d in dat:
            mpnn.forward(d[0], d[0].ndata['features'], d[0].edata['features'])
