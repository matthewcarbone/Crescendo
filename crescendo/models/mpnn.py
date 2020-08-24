#!/usr/bin/env python3

import torch.nn as nn

from dgllife.model import MPNNPredictor

from crescendo.models.embedding import Embedding


class MPNN(nn.Module):
    """Message passing neural network. This is formally a hetero-graph (a graph
    with node and edge features) -> fixed length vector machine learning
    algorithm."""

    def __init__(
        self,
        n_node_features,
        n_edge_features,
        output_size,
        n_node_embed=None,
        n_edge_embed=None,
        mpnn_args=dict()
    ):
        """Initializer.

        Parameters
        ----------
        n_node_features, n_edge_features : List[int]
            The number classes per features/node(edge). For example, if each
            node has two features, the first feature has 3 options and the
            second feature has 2 options, then n_node_features=[3, 2].
        output_size : int
            The length of the target vector.
        n_node_embed, n_edge_embed : int
            The user-chosen dimensions of the embedding layers. Defaults to
            one each.
        mpnn_args : dict
            Other args to pass to the MPNN. See the docs here:
            https://lifesci.dgl.ai/_modules/dgllife/model/model_zoo/
            mpnn_predictor.html
        """

        super().__init__()

        if n_node_embed is None:
            n_node_embed = [1 for _ in range(len(n_node_features))]
        if n_edge_embed is None:
            n_edge_embed = [1 for _ in range(len(n_edge_features))]

        self.embedding_h = Embedding(n_node_features, n_node_embed)
        self.embedding_e = Embedding(n_edge_features, n_edge_embed)

        self.mpnn = MPNNPredictor(
            node_in_feats=sum(n_node_embed), edge_in_feats=sum(n_edge_embed),
            n_tasks=output_size, **mpnn_args
        )

    def forward(self, g, h, e):
        """Forward prop for the MPNN. See the docs linked in the __init__ to
        see what forward does. The only extra step here is to perform the
        embedding before sending the nodes/edges to the MPNN."""

        h = self.embedding_h(h)
        e = self.embedding_e(e)
        return self.mpnn.forward(g, h, e)
