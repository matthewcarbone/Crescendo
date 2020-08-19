#!/usr/bin/env python3

import numpy as np

from crescendo.utils.graphs import graph_to_vector_dummy_dataset


class TestRandomGraphGenerator:

    def test_graph_to_vector_dummy_dataset(self):
        """This also tests the random_graph_generator which it makes useof."""

        dat = graph_to_vector_dummy_dataset(
            100, graph_max_size=10, graph_max_n_class=7,
            graph_max_e_class=5, target_size=4
        )

        assert len(dat) == 100
        assert all([g[0].number_of_nodes() <= 10 for g in dat])
        assert all([
            np.all(np.array(g[0].ndata['features']) < 7) for g in dat
        ])
        assert all([
            np.all(np.array(g[0].edata['features']) < 5) for g in dat
        ])
        targets = np.array([g[1] for g in dat])
        assert targets.shape == (100, 4)
        assert len(np.unique([g[2] for g in dat])) == 100
