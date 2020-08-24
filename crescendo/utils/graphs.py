#!/usr/bin/env python3

import random

from dgl import DGLGraph
import numpy as np
from scipy.sparse import coo_matrix
import torch


def random_graph_generator(max_size=10, max_n_class=7, max_e_class=5):
    """Generates a graph of random size, connectivity, node and edge features.
    We also have a minimum atoms requirement to ensure that we have at least
    one edge."""

    # Generate a random self-transpose binary matrix
    s = np.random.randint(low=2, high=max_size + 1)

    # This ensures that we have at least one bond in the molecule.
    while True:
        rand_mat = np.random.randint(low=0, high=2, size=(s, s))
        A = np.triu(rand_mat, k=1)
        if A.sum() > 0:
            break

    # Initialize the graph with the adjacency matrix
    g = DGLGraph()
    g.from_scipy_sparse_matrix(coo_matrix(A + A.T))

    # Add dummy features
    all_node_features = [
        [np.random.randint(low=0, high=max_n_class)]
        for nn in range(g.number_of_nodes())
    ]
    all_edge_features = [
        [np.random.randint(low=0, high=max_e_class)]
        for nn in range(g.number_of_edges())
    ]
    g.ndata['features'] = torch.LongTensor(all_node_features)
    g.edata['features'] = torch.LongTensor(all_edge_features)

    return g


def graph_to_vector_dummy_dataset(
    ds_size, graph_max_size=10, graph_max_n_class=7, graph_max_e_class=5,
    target_size=4
):
    """Generates a dummy graph dataset. This is intended to mimic the
    self.ml_data attribute generated in e.g. the qm9 dataset. The dataset is
    generated in a silly way so that the output target is correlated directly
    with the average node class."""

    ml_data = []
    for idx in range(ds_size):
        g = random_graph_generator(
            graph_max_size, graph_max_n_class, graph_max_e_class
        )
        mean_feature = g.ndata['features'].float().mean().item()

        # Our target is just the noisy mean of the feature classes
        target = np.ones(shape=target_size) * mean_feature + \
            np.random.normal(scale=0.1, size=target_size)

        ml_data.append([g, target, idx])

    random.shuffle(ml_data)
    return ml_data
