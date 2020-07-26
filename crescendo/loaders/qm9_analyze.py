#!/usr/bin/env python3

"""Module for analyzing data within QM9"""

from rdkit import Chem


class QM9Datum:
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
    >>> QM9Datum('C1=CC=CC=C1').aromatic
    True
    """
    
    ringpattern= Chem.MolFromSmarts('[r]')
    ring5pattern= Chem.MolFromSmarts('[r5]')
    ring4pattern= Chem.MolFromSmarts('[r4]')
    aromaticpattern=  Chem.MolFromSmarts('[a]')
    DBpattern1 = Chem.MolFromSmarts('C=C')
    DBpattern2 = Chem.MolFromSmarts('C=O')
    DBpattern3 = Chem.MolFromSmarts('C=N')
    DBpattern4 = Chem.MolFromSmarts('O=O')
    DBpattern5 = Chem.MolFromSmarts('O=N')
    DBpattern6 = Chem.MolFromSmarts('N=N')
    TBpattern1 = Chem.MolFromSmarts('C#C')
    TBpattern2 = Chem.MolFromSmarts('C#N')
    TBpattern3 = Chem.MolFromSmarts('N#N')
    

    def __init__(self, smiles):
        """
        Parameters
        ----------
        smiles : str
            smiles of target molecule as string
        """

        self.smile = smiles
        self.mol = Chem.MolFromSmiles(smiles)

    @property
    def ring(self):
        return self.mol.HasSubstructMatch(QM9Datum.ringpattern)

    @property
    def ring5(self):
        return self.mol.HasSubstructMatch(QM9Datum.ring5pattern)

    @property
    def ring4(self):
        return self.mol.HasSubstructMatch(QM9Datum.ring4pattern)

    @property
    def aromatic(self):
        return self.mol.HasSubstructMatch(QM9Datum.aromaticpattern)

    @property
    def doublebond(self):
        DBval1 = self.mol.HasSubstructMatch(QM9Datum.DBpattern1) or self.mol.HasSubstructMatch(QM9Datum.DBpattern2) or self.mol.HasSubstructMatch(QM9Datum.DBpattern3)
        DBval2=  self.mol.HasSubstructMatch(QM9Datum.DBpattern4) or self.mol.HasSubstructMatch(QM9Datum.DBpattern5) or self.mol.HasSubstructMatch(QM9Datum.DBpattern6) 
        DBval= DBval1 or DBval2
        return DBval

    @property
    def triplebond(self):
        TBval = self.mol.HasSubstructMatch(QM9Datum.TBpattern1) or self.mol.HasSubstructMatch(QM9Datum.TBpattern2) or self.mol.HasSubstructMatch(QM9Datum.TBpattern3)
        return TBval

    @property
    def singlebond(self):
        return not self.doublebond and not self.triplebond and not self.aromatic
