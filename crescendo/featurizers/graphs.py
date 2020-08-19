#!/usr/bin/env python3

"""Module for converting objects to graphs."""

from dgl import DGLGraph
from rdkit import Chem
import torch

# Note that for these features, 0 is reserved for an unknown type.

# Only some atoms are allowed in the QM9 database
atom_symbols_map = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5}

# Atom hybridization
hybridization_map = {
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3
}

# Bond types
bond_type_map = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3
}


def get_number_of_classes_per_feature(atom_feature_list, bond_feature_list):
    """A simple helper that returns the number of possible class options per
    choice of feature. For example for bond_types, there are actually four
    classes: single, double, triple and unknown. This helper simply indexes
    the number of choices for atom and bond features, as it is necessary
    for initializing the MPNN. The order of the entries in the list must
    correspond to that in mol_to_graph_via_DGL."""

    if atom_feature_list is None:
        node_options = [1]
    else:
        node_options = []
        if 'type' in atom_feature_list:
            node_options.append(len(atom_symbols_map) + 1)
        if 'hybridization' in atom_feature_list:
            node_options.append(len(hybridization_map) + 1)

    if bond_feature_list is None:
        edge_options = [1]
    else:
        edge_options = []
        if 'type' in bond_feature_list:
            edge_options.append(len(bond_type_map) + 1)

    return node_options, edge_options


def mol_to_graph_via_DGL(mol, atom_feature_list, bond_feature_list):
    """Converts a rdkit.Chem.mol object into a graph using DGL.

    Parameters
    ----------
    mol : rdkit.Chem.mol
        An rdkit molecule.
    atom_features, bond_features : List[str]
        A list of the desired atom.bond features. Can be None, in which case,
        each atom will get the same feature vector of [0].

    Returns
    -------
    DGLGraph
    """

    g = DGLGraph()
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    g.add_nodes(n_atoms)

    # In this initial stage, we construct the actual graph by connecting
    # nodes via the edges corresponding to molecular bonds.
    for bond_index in range(n_bonds):
        bond = mol.GetBondWithIdx(bond_index)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()

        # DGL graphs are by default *directed*. We make this an undirected
        # graph by adding "edges" in both directions, meaning u -> v and
        # v -> u.
        g.add_edges([u, v], [v, u])

    # Iterate through all nodes (atoms) and assign various features.
    all_node_features = []
    for atom_index in range(n_atoms):

        # In the case in which the user wants no atomic information specified,
        # we simply append the same value for every atom, which is the same
        # as including no information at all, but this way it is still
        # compatible with the DGL architecture.
        if atom_feature_list is None:
            all_node_features.append([0])
            continue

        atom_features = []
        atom = mol.GetAtomWithIdx(atom_index)

        # Append the atom type to the feature vector
        if 'type' in atom_feature_list:
            atom_features.append(atom_symbols_map.get(
                atom.GetSymbol(), 0)
            )

        # Append the atom hybridization to the feature vector
        if 'hybridization' in atom_feature_list:
            atom_features.append(hybridization_map.get(
                atom.GetHybridization(), 0)
            )

        all_node_features.append(atom_features)

    # Iterate through the bonds in an order consistent with the graph and
    # assign the edge features.
    all_edge_features = []
    for bond_index in range(n_bonds):

        if bond_feature_list is None:
            all_edge_features.extend([[0], [0]])
            continue

        bond_features = []
        bond = mol.GetBondWithIdx(bond_index)
        bond_type = bond_type_map.get(bond.GetBondType(), 0)

        if 'type' in bond_feature_list:
            bond_features.append(bond_type)

        # Need to add two since in order to create an undirected graph in DGL
        # you need edges pointing in both directions.
        all_edge_features.extend([bond_features, bond_features])

    if len(all_node_features) != 0:
        g.ndata['features'] = torch.LongTensor(all_node_features)
    if len(all_edge_features) != 0:
        g.edata['features'] = torch.LongTensor(all_edge_features)

    return g
