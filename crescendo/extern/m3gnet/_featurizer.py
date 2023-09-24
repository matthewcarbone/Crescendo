# Note that some of the code below is reproduced under the terms of the BSD
# 3-Clause License (specifically featurize_material).

# BSD 3-Clause License

# Copyright (c) 2022, Materials Virtual Lab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIALDAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import numpy as np

from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
)
from matgl.utils.cutoff import polynomial_cutoff
from matgl.ext.pymatgen import Structure2Graph
import torch

from functools import cache
from matgl import load_model


@cache
def _load_default_featurizer():
    model = load_model("M3GNet-MP-2021.2.8-PES").model
    model.eval()
    return model


def featurize_material(structure, model=_load_default_featurizer()):
    """Executes a featurization of a material using M3GNet.

    The following steps are performed:

    1. The model is placed in eval mode
    2. Graph is retrieved from the graph converter (Structure2Graph)
    3. Pair and vector distances are computed and appended to the graph
    features.
    4. A line graph is created
    5. Edges are applied from matgl.graph.compute.compute_theta_and_phi
    6. Basis expansion
    7. Polynomial cutoff
    8. Initial embedding
    9. Convolutions via M3GNet blocks

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
    model : matgl.models._m3gnet.M3GNet
        See notes below for required methods and attributes. This defaults to
        the pre-trained model 'M3GNet-MP-2021.2.8-PES'. You can load other
        models via ``matgl.load_model``.

    Returns
    -------
    np.ndarray
        The node features. Will have shape (number of atoms) x (number of
        hidden node features).

    Notes
    -----
    The provided model must have the following methods/attributes defined on
    it: bond_expansion(), threebody_cutoff, basis_expansion(),
    embedding(), n_blocks, three_body_interactions, graph_layers.
    """

    graph_converter = Structure2Graph(model.element_types, model.cutoff)
    g, state_attr = graph_converter.get_graph(structure)

    # Hacking the forward part of the model init -----
    node_types = g.ndata["node_type"]
    bond_vec, bond_dist = compute_pair_vector_and_distance(g)
    g.edata["bond_vec"] = bond_vec.to(g.device)
    g.edata["bond_dist"] = bond_dist.to(g.device)

    model.eval()

    with torch.no_grad():

        expanded_dists = model.bond_expansion(g.edata["bond_dist"])

        l_g = create_line_graph(g, model.threebody_cutoff)

        l_g.apply_edges(compute_theta_and_phi)
        g.edata["rbf"] = expanded_dists
        three_body_basis = model.basis_expansion(l_g)
        three_body_cutoff = polynomial_cutoff(
            g.edata["bond_dist"], model.threebody_cutoff
        )
        node_feat, edge_feat, state_feat = model.embedding(
            node_types, g.edata["rbf"], state_attr
        )

        for i in range(model.n_blocks):
            edge_feat = model.three_body_interactions[i](
                g,
                l_g,
                three_body_basis,
                three_body_cutoff,
                node_feat,
                edge_feat,
            )
            edge_feat, node_feat, state_feat = model.graph_layers[i](
                g, edge_feat, node_feat, state_feat
            )
    # Hacking the forward part of the model done -----

    return np.array(node_feat.detach().numpy())
