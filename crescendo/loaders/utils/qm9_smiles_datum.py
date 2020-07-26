#!/usr/bin/env python3

"""Module for analyzing data within QM9"""

from rdkit import Chem


class QM9SmilesDatum:
    """Functions for determining structures from a smiles input using rdkit
    chem.
    Current Structure types
        .ring -contains any ring
        .ring5 -contains a 5 atom ring
        .ring4 -contains a 4 atom ring
        .aromatic -contains an aromatic structure
        .doublebond -contains a double bond with the combinations of carbon,
        oxygen, and nitrogen
        .triplebond -contains a triple bond with the combinations of carbon
        and nitrogen
        .singlebond -does not contain .doublebond .triplebond and .aromatic

    Example
    -------
    #Molecule Benzene aromatic structure
    >>> d = QM9SmilesDatum('C1=CC=CC=C1')
    >>> d.is_aromatic()
    True
    """

    aromatic_pattern = Chem.MolFromSmarts('[a]')

    double_bond_patterns = [
        Chem.MolFromSmarts('C=C'), Chem.MolFromSmarts('C=O'),
        Chem.MolFromSmarts('C=N'), Chem.MolFromSmarts('O=O'),
        Chem.MolFromSmarts('O=N'), Chem.MolFromSmarts('N=N')
    ]

    triple_bond_patterns = [
        Chem.MolFromSmarts('C#C'), Chem.MolFromSmarts('C#N'),
        Chem.MolFromSmarts('N#N')
    ]

    def __init__(self, smiles):
        """
        Parameters
        ----------
        smiles : str
            smiles of target molecule as string
        """

        self.smile = smiles
        self.mol = Chem.MolFromSmiles(smiles)

    def has_n_membered_ring(self, n=None):
        """Returns True if the mol attribute (the molecule object in rdkit
        representing the Smiles string) has an n-membered ring.

        Parameters
        ----------
        n : int, optional
            The size of the ring. If None, will simply check if the molecule
            has any ring (using the substructure matching string '[r]' instead
            of [f'r{n}']). Default is None.

        Returns
        -------
        bool
            True if a match is found. False otherwise.
        """

        n = '' if n is None else n
        return self.mol.HasSubstructMatch(Chem.MolFromSmarts(f'[r{n}]'))

    def is_aromatic(self):
        """If the molecule has any aromatic pattern, returns True, else
        returns False. This is checked by trying to locate the substructure
        match for the string '[a]'.

        Returns
        -------
        bool
        """

        return self.mol.HasSubstructMatch(QM9SmilesDatum.aromatic_pattern)

    def has_double_bond(self):
        """Checks the molecule for double bonds. Note that by default this
        method assumes the data point is from QM9, and only checks the
        atoms capable of forming double bonds in QM9, so, it will only check
        C, N and O, and their 6 combinations.

        Returns
        -------
        bool
        """

        return any([
            self.mol.HasSubstructMatch(p)
            for p in QM9SmilesDatum.double_bond_patterns
        ])

    def has_triple_bond(self):
        """Checks the molecule for triple bonds. Note that by default this
        method assumes the data point is from QM9, and only checks the
        atoms capable of forming double bonds in QM9, so, it will only check
        C#C, C#N and N#N.

        Returns
        -------
        bool
        """

        return any([
            self.mol.HasSubstructMatch(p)
            for p in QM9SmilesDatum.triple_bond_patterns
        ])
