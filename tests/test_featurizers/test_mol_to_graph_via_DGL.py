#!/usr/bin/env python3

import pytest

from crescendo.datasets.qm9 import QMXDataset
from crescendo.featurizers.graphs import mol_to_graph_via_DGL, \
    atom_symbols_map, hybridization_map, bond_type_map, \
    get_number_of_classes_per_feature


"""Useful information about what the keys are

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
"""

# Use the QMXDataset to load in the testing data
# qm9_ids_in_testing_data = [
#     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100001, 100002, 100003, 100004
# ]

# Atom/bond feature lists (defaults)
AFL = ['type', 'hybridization']
BFL = ['type']

N_ATOM_TYPE = len(atom_symbols_map)
N_HYBRID_TYPE = len(hybridization_map)
N_BOND_TYPE = len(bond_type_map)


@pytest.fixture
def data():
    ds = QMXDataset()
    ds.load("data/qm9_test_data")
    return ds.raw


class TestAuxiliary:

    def test_get_number_of_classes_per_feature(self):
        nf = get_number_of_classes_per_feature(None, None)
        node_options = nf[0]
        edge_options = nf[1]
        assert node_options == [1]
        assert edge_options == [1]
        nf = get_number_of_classes_per_feature(AFL, BFL)
        node_options = nf[0]
        edge_options = nf[1]

        # Each option allows for an "unknown" class
        assert node_options[0] == N_ATOM_TYPE + 1
        assert node_options[1] == N_HYBRID_TYPE + 1
        assert edge_options[0] == N_BOND_TYPE + 1


class TestMolToGraphViaDGLNoFeatures:

    def test_000009(self, data):

        graph = mol_to_graph_via_DGL(data[9].mol, None, None)
        assert graph.ndata['features'][0][0] == 0
        assert graph.ndata['features'][1][0] == 0
        assert graph.ndata['features'][2][0] == 0

        assert graph.edata['features'][0][0] == 0
        assert graph.edata['features'][1][0] == 0
        assert graph.edata['features'][2][0] == 0
        assert graph.edata['features'][3][0] == 0


class TestMolToGraphViaDGL:

    def test_000001(self, data):
        """CH4 - note hydrogen atoms are not counted in the graph.
        * The carbon atom is of type 2, with hybridization type 3
        * The bond information is null (no non-C-H bonds)
        """

        graph = mol_to_graph_via_DGL(data[1].mol, AFL, BFL)
        assert graph.ndata['features'][0][0] == 2
        assert graph.ndata['features'][0][1] == 3
        assert graph.edata == dict()
        assert len(graph.edata) == 0

    def test_000002(self, data):
        """NH3
        * The nitrogen atom is of type 3, with hybridization of type 3 (don't
        forget the lone pair!)
        * The bond information is null.
        """

        graph = mol_to_graph_via_DGL(data[2].mol, AFL, BFL)
        assert graph.ndata['features'][0][0] == 3
        assert graph.ndata['features'][0][1] == 3
        assert graph.edata == dict()
        assert len(graph.edata) == 0

    def test_000003(self, data):
        """OH2 (water)
        * The oxygen atom is of type 4, with hybridization of type 3 (don't
        forget the two lone pairs!)
        * The bond information is null.
        """

        graph = mol_to_graph_via_DGL(data[3].mol, AFL, BFL)
        assert graph.ndata['features'][0][0] == 4
        assert graph.ndata['features'][0][1] == 3
        assert graph.edata == dict()
        assert len(graph.edata) == 0

    def test_000004(self, data):
        """C#C (carbon triple-bonded carbon). We now have a more complicated
        molecule to test.
        * Each carbon atom is of type 2, with hybridizations of type 1.
        * There is a single triple bond between the atoms, so we should have
        TWO edges of type 3, since don't forget, we need a directed edge in
        each direction.
        """

        graph = mol_to_graph_via_DGL(data[4].mol, AFL, BFL)
        assert graph.ndata['features'][0][0] == 2
        assert graph.ndata['features'][1][0] == 2
        assert graph.ndata['features'][0][1] == 1
        assert graph.ndata['features'][1][1] == 1
        assert graph.edata['features'][0][0] == 3
        assert graph.edata['features'][1][0] == 3

    def test_000005(self, data):
        """C#N (carbon triple-bonded nitrogen)
        * The carbon atom is of type 2, with hybridization of type 1.
        * Same for the nitrogen atom, except it is of type 3.
        * There is a single triple bond between the atoms, so we need two edges
        of type 3.
        """

        graph = mol_to_graph_via_DGL(data[5].mol, AFL, BFL)
        assert graph.ndata['features'][0][0] == 2
        assert graph.ndata['features'][1][0] == 3
        assert graph.ndata['features'][0][1] == 1
        assert graph.ndata['features'][1][1] == 1
        assert graph.edata['features'][0][0] == 3
        assert graph.edata['features'][1][0] == 3

    def test_000006(self, data):
        """C=O (carbon double-bonded oxygen)
        * The carbon atom is of type 2, with hybridization of type 2.
        * The oxygen atom is of type 4, with hybridization of type 2.
        * There is a single double bond (type 2) between them.
        """

        graph = mol_to_graph_via_DGL(data[6].mol, AFL, BFL)
        assert graph.ndata['features'][0][0] == 2
        assert graph.ndata['features'][1][0] == 4
        assert graph.ndata['features'][0][1] == 2
        assert graph.ndata['features'][1][1] == 2
        assert graph.edata['features'][0][0] == 2
        assert graph.edata['features'][1][0] == 2

    def test_000007(self, data):
        """C-C (carbon single-bonded carbon)
        * The carbon atoms are of type 2, with hybridizations of type 3.
        * There is a single single bond between them, of type 1.
        """

        graph = mol_to_graph_via_DGL(data[7].mol, AFL, BFL)
        assert graph.ndata['features'][0][0] == 2
        assert graph.ndata['features'][1][0] == 2
        assert graph.ndata['features'][0][1] == 3
        assert graph.ndata['features'][1][1] == 3
        assert graph.edata['features'][0][0] == 1
        assert graph.edata['features'][1][0] == 1

    def test_000009(self, data):
        """CC#C
        * The indexing starts at the carbon atom which is SP3 hybridized.
        * The first bond is a single bond.
        * The second bond is a triple bond.
        """

        graph = mol_to_graph_via_DGL(data[9].mol, AFL, BFL)
        assert graph.ndata['features'][0][0] == 2
        assert graph.ndata['features'][1][0] == 2
        assert graph.ndata['features'][2][0] == 2

        assert graph.ndata['features'][0][1] == 3
        assert graph.ndata['features'][1][1] == 1
        assert graph.ndata['features'][2][1] == 1

        assert graph.edata['features'][0][0] == 1
        assert graph.edata['features'][1][0] == 1
        assert graph.edata['features'][2][0] == 3
        assert graph.edata['features'][3][0] == 3
