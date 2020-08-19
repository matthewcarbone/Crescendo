#!/usr/bin/env python3

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """Based off of code in
    https://github.com/graphdeeplearning/benchmarking-gnns"""

    def __init__(self, input_dims, embedding_dims, padding_index=0):
        """Embedding module for embedding e.g. a list of node features into
        a learned embedding layer. For example, consider that each atom in
        a molecule has two features. The first is the atom type for which there
        are 4 defined options, and the second is the hybridization for which
        there are 3 options. In this case, assuming we allow for an "unknown"
        class option, the input_dims = [5, 4]. The embedding_dims must be
        of the same length, but the user can choose the dimension of the
        embedding, e.g. [10, 7].

        Parameters
        ----------
        input_dims : List[int]
            Number of classes for each feature.
        embedding_dims : List[int]
            Embedding length for each feature. For instance, if
            embedding_dims=[3, 2], then feature[0] has embedding length 3, and
            feature[1] has embedding length 2.
        """

        super().__init__()
        assert len(input_dims) == len(embedding_dims)

        self.layers = nn.ModuleList(
            nn.Embedding(feature_dim, embedding_dim, padding_idx=padding_index)
            for feature_dim, embedding_dim in zip(input_dims, embedding_dims)
        )

    def forward(self, features):
        """Forward prop.

        Parameters
        ----------
        features : torch.tensor
            Of dimension [batch size, feature size].

        Returns
        -------
        torch.tensor
            Of dimension [batch size, sum(self.embedding_dims)]
        """

        return torch.cat([
            network(features[:, ii]) for ii, network in enumerate(self.layers)
        ], dim=1)
